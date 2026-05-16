//
//  SpeechRecognizer.swift
//  Textream
//
//  Created by Fatih Kadir Akın on 8.02.2026.
//

import AppKit
import Foundation
import Speech
import AVFoundation
import CoreAudio

struct AudioInputDevice: Identifiable, Hashable {
    let id: AudioDeviceID
    let uid: String
    let name: String

    static func allInputDevices() -> [AudioInputDevice] {
        var propertyAddress = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDevices,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var dataSize: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(AudioObjectID(kAudioObjectSystemObject), &propertyAddress, 0, nil, &dataSize) == noErr else { return [] }

        let deviceCount = Int(dataSize) / MemoryLayout<AudioDeviceID>.size
        var deviceIDs = [AudioDeviceID](repeating: 0, count: deviceCount)
        guard AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &propertyAddress, 0, nil, &dataSize, &deviceIDs) == noErr else { return [] }

        var result: [AudioInputDevice] = []
        for deviceID in deviceIDs {
            // Check if device has input streams
            var inputAddress = AudioObjectPropertyAddress(
                mSelector: kAudioDevicePropertyStreams,
                mScope: kAudioDevicePropertyScopeInput,
                mElement: kAudioObjectPropertyElementMain
            )
            var streamSize: UInt32 = 0
            guard AudioObjectGetPropertyDataSize(deviceID, &inputAddress, 0, nil, &streamSize) == noErr, streamSize > 0 else { continue }

            // Get UID
            var uidAddress = AudioObjectPropertyAddress(
                mSelector: kAudioDevicePropertyDeviceUID,
                mScope: kAudioObjectPropertyScopeGlobal,
                mElement: kAudioObjectPropertyElementMain
            )
            var uid: CFString = "" as CFString
            var uidSize = UInt32(MemoryLayout<CFString>.size)
            guard AudioObjectGetPropertyData(deviceID, &uidAddress, 0, nil, &uidSize, &uid) == noErr else { continue }

            // Get name
            var nameAddress = AudioObjectPropertyAddress(
                mSelector: kAudioDevicePropertyDeviceNameCFString,
                mScope: kAudioObjectPropertyScopeGlobal,
                mElement: kAudioObjectPropertyElementMain
            )
            var name: CFString = "" as CFString
            var nameSize = UInt32(MemoryLayout<CFString>.size)
            guard AudioObjectGetPropertyData(deviceID, &nameAddress, 0, nil, &nameSize, &name) == noErr else { continue }

            result.append(AudioInputDevice(id: deviceID, uid: uid as String, name: name as String))
        }
        return result
    }

    static func deviceID(forUID uid: String) -> AudioDeviceID? {
        allInputDevices().first(where: { $0.uid == uid })?.id
    }
}

@Observable
class SpeechRecognizer {
    var recognizedCharCount: Int = 0
    var isListening: Bool = false
    var error: String?
    var audioLevels: [CGFloat] = Array(repeating: 0, count: 30)
    var lastSpokenText: String = ""
    var shouldDismiss: Bool = false
    var shouldAdvancePage: Bool = false

    /// True when recent audio levels indicate the user is actively speaking
    var isSpeaking: Bool {
        let recent = audioLevels.suffix(10)
        guard !recent.isEmpty else { return false }
        let avg = recent.reduce(0, +) / CGFloat(recent.count)
        return avg > 0.08
    }

    private var speechRecognizer: SFSpeechRecognizer?
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private var audioEngine = AVAudioEngine()
    private var sourceText: String = ""
    private var normalizedSource: String = ""
    private var matchStartOffset: Int = 0  // char offset to start matching from
    private var retryCount: Int = 0
    private let maxRetries: Int = 10
    private var configurationChangeObserver: Any?
    private var pendingRestart: DispatchWorkItem?
    private var sessionGeneration: Int = 0
    private var suppressConfigChange: Bool = false
    private var requestLock = NSLock()
    private var preemptiveRestartTimer: Timer?
    /// Sliding window of recent match positions for confidence gating.
    /// We require 2-of-3 recent results to agree before committing a forward jump.
    private var recentMatchPositions: [Int] = []

    // MARK: - 意译识别（本地语义兜底层）
    //
    // 旧逐字/逐词匹配拿不准时，用本地语义判断「用户说的话和稿子前方哪一段意思最近」。
    // 旧逻辑高置信命中时直接返回，语义不否决（向后兼容、降回归风险）。
    // 调参最终值由 TASK-003 在 knowledge/generated/spike-semantic 用更大样本敲定后写入
    // knowledge/decisions.md；此处为 PROJECT-SPEC/decisions 的候选默认值。
    private let semanticMatcher = SemanticMatcher()
    /// 语义层是否就绪（主线程读写）。模型未就绪期间纯走旧逻辑，不阻塞跟读。
    private var semanticReady = false
    /// 串行队列：NLContextualEmbedding 未声明线程安全，编码在此串行、不占主线程。
    private let semanticQueue = DispatchQueue(label: "com.textream.semantic", qos: .userInitiated)
    /// 防止异步语义任务在主线程间堆叠。
    private var semanticInFlight = false
    /// 主阈值 T：≥T 视为语义强匹配（冻结操作阈值，见 decisions.md 第 1 条）。
    private let semanticThreshold = 0.70
    /// 不确定带宽 δ：[T-δ, T) 不推进（极端意译留给 TASK-005 云端补漏）。
    private let semanticUncertainBand = 0.10
    /// 仅向前候选段数 K：只与指针前方至多 K 段比对，天然不向后匹配。
    private let semanticForwardSegments = 3
    /// 语义窗口上限：SFSpeechRecognizer 单次会话的 formattedString 是「本会话累计转写」，
    /// 中文连续听写常不吐句末标点，靠标点切「最近一句」会退化成整段累计文本，
    /// 语义向量被稀释、后半段相似度跌破 T 而停止推进。故窗口强制只取结尾一小段。
    /// 此长度为候选默认值，最终由 TASK-003 调参定值。
    private let semanticWindowMaxChars = 48
    /// 单次语义确认后最多前进的段数（粒度细化）：语义命中前方第 i 段时，原逻辑一次推进
    /// 过 0...i 全部段（最多 K 段一跳，体感"窜一大段"）；此上限把单次推进收敛到至多
    /// N 段，靠后续连续匹配逐段追上，更接近逐句推进。仅缩小"一次走多远"，不改匹配判定
    /// 与 2/3 防误翻门禁——走得更少更保守、更偏「宁可漏翻」，防误翻只增不减。
    /// 候选默认 1，最终由 TASK-003 调参定值（与 T/δ/K/M/窗口/静默 同批）。
    private let semanticMaxAdvanceSegments = 1
    /// 配速锚 / 领先夹紧上限：指针相对「matchStartOffset + 本会话累计转写经
    /// splitTextIntoWords 投影后的脚本单位量」最多领先的字符数。超出即不推进，
    /// 防止旧逐字/逐词在意译或只说几个词时窜到稿子前方 3-4 句、埋掉未读行。
    /// 用户裁决=严格锁定：取紧的小余量、强偏「宁可漏翻不可误翻」。
    /// 候选默认 8，最终由 TASK-003 调参定值（与 T/δ/K/M/窗口/推进段数 同批）。
    private let semanticMaxLeadChars = 8

    /// Update the source text while preserving the current recognized char count.
    /// Used by Director Mode to live-edit unread text without resetting read progress.
    func updateText(_ text: String, preservingCharCount: Int) {
        let words = splitTextIntoWords(text)
        let collapsed = words.joined(separator: " ")
        sourceText = collapsed
        normalizedSource = Self.normalize(collapsed)
        recognizedCharCount = min(preservingCharCount, collapsed.count)
        matchStartOffset = recognizedCharCount
        recentMatchPositions = []
    }

    /// Jump highlight to a specific char offset (e.g. when user taps a word)
    func jumpTo(charOffset: Int) {
        recognizedCharCount = charOffset
        matchStartOffset = charOffset
        retryCount = 0
        recentMatchPositions = []
        if isListening {
            restartRecognition()
        }
    }

    func start(with text: String) {
        // Clean up any previous session immediately so pending restarts
        // and stale taps are removed before the async auth callback fires.
        cleanupRecognition()

        let words = splitTextIntoWords(text)
        let collapsed = words.joined(separator: " ")
        sourceText = collapsed
        normalizedSource = Self.normalize(collapsed)
        recognizedCharCount = 0
        matchStartOffset = 0
        retryCount = 0
        recentMatchPositions = []
        error = nil
        sessionGeneration += 1
        prepareSemanticIfNeeded()

        // Check microphone permission first
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .denied, .restricted:
            error = "Microphone access denied. Open System Settings → Privacy & Security → Microphone to allow Textream."
            openMicrophoneSettings()
            return
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .audio) { [weak self] granted in
                DispatchQueue.main.async {
                    if granted {
                        self?.requestSpeechAuthAndBegin()
                    } else {
                        self?.error = "Microphone access denied. Open System Settings → Privacy & Security → Microphone to allow Textream."
                    }
                }
            }
            return
        case .authorized:
            break
        @unknown default:
            break
        }

        requestSpeechAuthAndBegin()
    }

    private func requestSpeechAuthAndBegin() {
        SFSpeechRecognizer.requestAuthorization { [weak self] status in
            DispatchQueue.main.async {
                switch status {
                case .authorized:
                    self?.beginRecognition()
                default:
                    self?.error = "Speech recognition not authorized. Open System Settings → Privacy & Security → Speech Recognition to allow Textream."
                    self?.openSpeechRecognitionSettings()
                }
            }
        }
    }

    private func openMicrophoneSettings() {
        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone") {
            NSWorkspace.shared.open(url)
        }
    }

    private func openSpeechRecognitionSettings() {
        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_SpeechRecognition") {
            NSWorkspace.shared.open(url)
        }
    }

    func stop() {
        isListening = false
        cleanupRecognition()
    }

    func forceStop() {
        isListening = false
        sourceText = ""
        retryCount = maxRetries
        recentMatchPositions = []
        cleanupRecognition()
    }

    func resume() {
        retryCount = 0
        matchStartOffset = recognizedCharCount
        recentMatchPositions = []
        shouldDismiss = false
        prepareSemanticIfNeeded()
        beginRecognition()
    }

    /// 在非主线程一次性加载本地语义模型资产。离线/无资产时保持降级（semanticReady=false），
    /// 不抛错、不阻塞「开始跟读」。已就绪则跳过。
    private func prepareSemanticIfNeeded() {
        if semanticReady { return }
        semanticQueue.async { [weak self] in
            guard let self else { return }
            let ok = self.semanticMatcher.prepare()
            DispatchQueue.main.async { self.semanticReady = ok }
        }
    }

    private func cleanupRecognitionTask() {
        // Cancel any pending restart to prevent overlapping beginRecognition calls
        pendingRestart?.cancel()
        pendingRestart = nil

        stopPreemptiveTimer()

        if let observer = configurationChangeObserver {
            NotificationCenter.default.removeObserver(observer)
            configurationChangeObserver = nil
        }
        requestLock.lock()
        recognitionRequest?.endAudio()
        recognitionRequest = nil
        requestLock.unlock()
        recognitionTask?.cancel()
        recognitionTask = nil
    }

    private func cleanupAudioEngine() {
        if audioEngine.isRunning {
            audioEngine.stop()
        }
        audioEngine.inputNode.removeTap(onBus: 0)
    }

    private func cleanupRecognition() {
        cleanupRecognitionTask()
        cleanupAudioEngine()
    }

    /// Coalesces all delayed beginRecognition() calls into a single pending work item.
    /// Any previously scheduled restart is cancelled before the new one is queued.
    private func scheduleBeginRecognition(after delay: TimeInterval) {
        pendingRestart?.cancel()
        let work = DispatchWorkItem { [weak self] in
            guard let self else { return }
            self.pendingRestart = nil
            self.beginRecognition()
        }
        pendingRestart = work
        DispatchQueue.main.asyncAfter(deadline: .now() + delay, execute: work)
    }

    private func beginRecognition() {
        // Ensure clean state
        cleanupRecognition()

        // Create a fresh engine so it picks up the current hardware format.
        // AVAudioEngine caches the device format internally and reset() alone
        // does not reliably flush it after a mic switch.
        audioEngine = AVAudioEngine()

        // Set selected microphone if configured
        let micUID = NotchSettings.shared.selectedMicUID
        if !micUID.isEmpty, let deviceID = AudioInputDevice.deviceID(forUID: micUID) {
            // Suppress config-change observer during our own device switch
            suppressConfigChange = true
            let inputUnit = audioEngine.inputNode.audioUnit
            if let audioUnit = inputUnit {
                var devID = deviceID
                AudioUnitSetProperty(
                    audioUnit,
                    kAudioOutputUnitProperty_CurrentDevice,
                    kAudioUnitScope_Global,
                    0,
                    &devID,
                    UInt32(MemoryLayout<AudioDeviceID>.size)
                )
                // Re-initialize audio unit so it picks up the new device's format
                AudioUnitUninitialize(audioUnit)
                AudioUnitInitialize(audioUnit)
            }
            // Allow config changes again after a settle period
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) { [weak self] in
                self?.suppressConfigChange = false
            }
        }

        speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: NotchSettings.shared.speechLocale))
        guard let speechRecognizer, speechRecognizer.isAvailable else {
            error = "Speech recognizer not available"
            return
        }

        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        guard let recognitionRequest else { return }
        recognitionRequest.shouldReportPartialResults = true

        // Add contextual strings from the source text to improve STT accuracy
        let upcoming = String(sourceText.dropFirst(matchStartOffset))
        let contextWords = upcoming.split(separator: " ")
            .map { String($0).lowercased().filter { $0.isLetter || $0.isNumber } }
            .filter { $0.count >= 5 }
        let uniqueContextWords = Array(Set(contextWords).prefix(50))
        if !uniqueContextWords.isEmpty {
            recognitionRequest.contextualStrings = uniqueContextWords
        }

        let inputNode = audioEngine.inputNode
        let hardwareFormat = inputNode.outputFormat(forBus: 0)

        // Guard against invalid format during device transitions (e.g. mic switch)
        guard hardwareFormat.sampleRate > 0, hardwareFormat.channelCount > 0 else {
            // Retry after a longer delay to let the audio system settle
            if retryCount < maxRetries {
                retryCount += 1
                scheduleBeginRecognition(after: 0.5)
            } else {
                error = "Audio input unavailable"
                isListening = false
            }
            return
        }

        // SFSpeechRecognizer requires mono audio. Multi-channel devices (e.g.
        // RODECaster Pro II at 2ch/48kHz) cause the recognition task to silently
        // return no results. Request a mono tap and let AVAudioEngine downmix.
        let monoFormat = AVAudioFormat(
            commonFormat: hardwareFormat.commonFormat,
            sampleRate: hardwareFormat.sampleRate,
            channels: 1,
            interleaved: hardwareFormat.isInterleaved
        )
        let tapFormat = (hardwareFormat.channelCount > 1) ? monoFormat : hardwareFormat

        // Observe audio configuration changes (e.g. mic switched externally) to restart gracefully
        configurationChangeObserver = NotificationCenter.default.addObserver(
            forName: .AVAudioEngineConfigurationChange,
            object: audioEngine,
            queue: .main
        ) { [weak self] _ in
            guard let self, !self.suppressConfigChange, !self.sourceText.isEmpty else { return }
            self.restartRecognition()
        }

        // Belt-and-suspenders: ensure no stale tap exists before installing
        inputNode.removeTap(onBus: 0)

        inputNode.installTap(onBus: 0, bufferSize: 1024, format: tapFormat) { [weak self] buffer, _ in
            self?.appendBufferToRequest(buffer)

            guard let channelData = buffer.floatChannelData?[0] else { return }
            let frameLength = Int(buffer.frameLength)
            var sum: Float = 0
            for i in 0..<frameLength {
                sum += channelData[i] * channelData[i]
            }
            let rms = sqrt(sum / Float(max(frameLength, 1)))
            let level = CGFloat(min(rms * 5, 1.0))

            DispatchQueue.main.async {
                self?.audioLevels.append(level)
                if (self?.audioLevels.count ?? 0) > 30 {
                    self?.audioLevels.removeFirst()
                }
            }
        }

        let currentGeneration = sessionGeneration
        recognitionTask = speechRecognizer.recognitionTask(with: recognitionRequest) { [weak self] result, error in
            guard let self else { return }
            if let result {
                let spoken = result.bestTranscription.formattedString
                DispatchQueue.main.async {
                    // Ignore stale results from a previous session
                    guard self.sessionGeneration == currentGeneration else { return }
                    self.retryCount = 0 // Reset on success
                    self.lastSpokenText = spoken
                    self.matchCharacters(spoken: spoken)
                }
            }
            if let error {
                DispatchQueue.main.async {
                    // If recognitionRequest is nil, cleanup already ran (intentional cancel) — don't retry
                    guard self.recognitionRequest != nil else { return }
                    guard self.isListening && !self.shouldDismiss && !self.sourceText.isEmpty else {
                        self.isListening = false
                        return
                    }

                    self.matchStartOffset = self.recognizedCharCount

                    // Distinguish timeout errors (expected every ~60s) from real errors.
                    // SFSpeechRecognizer timeout is error code 1110 in kAFAssistantErrorDomain,
                    // or 216 (kAudioConverterErr_FormatNotSupported). Retry immediately for
                    // timeouts with no retry limit; use backoff for real errors.
                    let nsError = error as NSError
                    let isTimeout = nsError.code == 1110 || nsError.code == 216

                    if isTimeout {
                        // Expected timeout — restart immediately, no retry limit
                        self.retryCount = 0
                        if self.audioEngine.isRunning {
                            self.restartTask()
                        } else {
                            self.scheduleBeginRecognition(after: 0.1)
                        }
                    } else if self.retryCount < self.maxRetries {
                        self.retryCount += 1
                        let delay = min(Double(self.retryCount) * 0.5, 1.5)
                        self.scheduleBeginRecognition(after: delay)
                    } else {
                        self.isListening = false
                    }
                }
            }
        }

        do {
            audioEngine.prepare()
            try audioEngine.start()
            isListening = true
            startPreemptiveTimer()
        } catch {
            // Transient failure after a device switch — retry with longer delay
            if retryCount < maxRetries {
                retryCount += 1
                scheduleBeginRecognition(after: 0.5)
            } else {
                self.error = "Audio engine failed: \(error.localizedDescription)"
                isListening = false
            }
        }
    }

    private func restartRecognition() {
        retryCount = 0
        isListening = true
        if audioEngine.isRunning {
            restartTask()
        } else {
            cleanupRecognition()
            scheduleBeginRecognition(after: 0.5)
        }
    }

    // MARK: - Thread-safe buffer appending

    private func appendBufferToRequest(_ buffer: AVAudioPCMBuffer) {
        requestLock.lock()
        recognitionRequest?.append(buffer)
        requestLock.unlock()
    }

    // MARK: - Soft restart (task only, keeps audio engine running)

    private func restartTask() {
        // Update match offset before restarting
        matchStartOffset = recognizedCharCount
        recentMatchPositions = []

        // Cancel any pending restart to avoid stale beginRecognition clobbering this session
        pendingRestart?.cancel()
        pendingRestart = nil

        // Cancel the old task and atomically swap to a new request under lock.
        // The lock prevents the audio tap from appending to the old request
        // between endAudio() and the new assignment.
        let newRequest = SFSpeechAudioBufferRecognitionRequest()
        newRequest.shouldReportPartialResults = true

        // Add contextual strings for the remaining text
        let upcoming = String(sourceText.dropFirst(matchStartOffset))
        let contextWords = upcoming.split(separator: " ")
            .map { String($0).lowercased().filter { $0.isLetter || $0.isNumber } }
            .filter { $0.count >= 5 }
        let uniqueWords = Array(Set(contextWords).prefix(50))
        if !uniqueWords.isEmpty {
            newRequest.contextualStrings = uniqueWords
        }

        // Nil out recognitionRequest before cancelling the old task so the
        // old task's error callback sees nil and skips retry logic. Then set
        // the new request after cancellation.
        requestLock.lock()
        recognitionRequest?.endAudio()
        recognitionRequest = nil
        requestLock.unlock()
        recognitionTask?.cancel()
        recognitionTask = nil

        requestLock.lock()
        recognitionRequest = newRequest
        requestLock.unlock()

        // Start new recognition task
        guard let speechRecognizer, speechRecognizer.isAvailable else {
            error = "Speech recognizer not available"
            isListening = false
            return
        }

        let currentGeneration = sessionGeneration
        recognitionTask = speechRecognizer.recognitionTask(with: newRequest) { [weak self] result, error in
            guard let self else { return }
            if let result {
                let spoken = result.bestTranscription.formattedString
                DispatchQueue.main.async {
                    guard self.sessionGeneration == currentGeneration else { return }
                    self.retryCount = 0
                    self.lastSpokenText = spoken
                    self.matchCharacters(spoken: spoken)
                }
            }
            if let error {
                DispatchQueue.main.async {
                    guard self.recognitionRequest != nil else { return }
                    guard self.isListening && !self.shouldDismiss && !self.sourceText.isEmpty else {
                        self.isListening = false
                        return
                    }

                    self.matchStartOffset = self.recognizedCharCount

                    let nsError = error as NSError
                    let isTimeout = nsError.code == 1110 || nsError.code == 216

                    if isTimeout {
                        self.retryCount = 0
                        if self.audioEngine.isRunning {
                            self.restartTask()
                        } else {
                            self.scheduleBeginRecognition(after: 0.1)
                        }
                    } else if self.retryCount < self.maxRetries {
                        self.retryCount += 1
                        let delay = min(Double(self.retryCount) * 0.5, 1.5)
                        self.scheduleBeginRecognition(after: delay)
                    } else {
                        self.isListening = false
                    }
                }
            }
        }

        startPreemptiveTimer()
    }

    // MARK: - Pre-emptive restart timer

    private func startPreemptiveTimer() {
        preemptiveRestartTimer?.invalidate()
        preemptiveRestartTimer = Timer.scheduledTimer(withTimeInterval: 55.0, repeats: true) { [weak self] _ in
            guard let self, self.isListening, !self.sourceText.isEmpty else { return }
            self.restartTask()
        }
    }

    private func stopPreemptiveTimer() {
        preemptiveRestartTimer?.invalidate()
        preemptiveRestartTimer = nil
    }

    // MARK: - Fuzzy character-level matching

    private func matchCharacters(spoken: String) {
        // Strategy 1: character-level fuzzy match from the start offset
        let charResult = charLevelMatch(spoken: spoken)

        // Strategy 2: word-level match (handles STT word substitutions)
        let wordResult = wordLevelMatch(spoken: spoken)

        // Use agreement-based selection instead of blind max().
        // If both strategies agree within a tolerance, use the average.
        // If they disagree wildly, use the more conservative (lower) result
        // to avoid false-positive jumps.
        let best: Int
        let tolerance = 20 // characters
        if abs(charResult - wordResult) <= tolerance {
            best = (charResult + wordResult) / 2
        } else {
            best = min(charResult, wordResult)
        }

        let newCount = matchStartOffset + best
        var legacyCommitted = false

        if newCount > recognizedCharCount {
            // 配速锚：candidate 不得超过「matchStartOffset + 已说出脚本单位量 + 余量」，
            // 防止旧逐字/逐词窜到稿子前方埋掉未读行。smallStep 由夹紧后 candidate 计算，
            // ≤15 小步因此无法链式越过此天花板（结构性闭合 smallStep 漏洞）。
            let paceCeiling = matchStartOffset + spokenScriptUnits(spoken) + semanticMaxLeadChars
            let candidate = min(newCount, sourceText.count, paceCeiling)

            // Confidence gating: require 2-of-3 recent results to agree on
            // forward movement to avoid single-result false-positive jumps.
            recentMatchPositions.append(candidate)
            if recentMatchPositions.count > 3 {
                recentMatchPositions.removeFirst()
            }

            // Check if at least 2 of the recent positions agree (within tolerance)
            let agreementThreshold = 10 // characters
            var confirmed = false
            if recentMatchPositions.count >= 2 {
                var agreeCount = 0
                for pos in recentMatchPositions {
                    if abs(pos - candidate) <= agreementThreshold {
                        agreeCount += 1
                    }
                }
                confirmed = agreeCount >= 2
            }

            // Small forward movements (< 1 word length) are always allowed
            // to keep the highlight responsive for normal reading
            let smallStep = candidate - recognizedCharCount <= 15

            if confirmed || smallStep {
                recognizedCharCount = candidate
                legacyCommitted = true
            }
        }

        // 语义兜底：仅当旧逐字/逐词本次未确认前进（拿不准或无进展）时启用。
        // 旧逻辑高置信命中时上面已 commit，这里直接 return，语义不否决。
        if legacyCommitted { return }
        attemptSemanticMatch(spoken: spoken)
    }

    // MARK: - 语义兜底匹配

    /// 旧逻辑拿不准时调用：把「最近口语窗口」与「指针前方至多 K 段」做本地语义比对，
    /// 异步编码不占主线程；结果回主线程并入同一 2/3 确认门禁，过期则丢弃（防误翻/防抖动）。
    private func attemptSemanticMatch(spoken: String) {
        guard semanticReady, !semanticInFlight else { return }
        guard recognizedCharCount < sourceText.count else { return }

        let window = recentSpokenWindow(spoken)
        guard window.count >= 2 else { return }

        let pointerSnapshot = recognizedCharCount
        let generationSnapshot = sessionGeneration
        let segments = forwardSemanticSegments(fromOffset: pointerSnapshot,
                                               maxCount: semanticForwardSegments)
        guard !segments.isEmpty else { return }

        semanticInFlight = true
        let candidateTexts = segments.map { $0.text }
        semanticQueue.async { [weak self] in
            guard let self else { return }
            let result = self.semanticMatcher.bestForwardMatch(
                spokenWindow: window,
                forwardCandidates: candidateTexts
            )
            DispatchQueue.main.async {
                self.semanticInFlight = false

                // 过期丢弃：会话已切换，或指针已被后续旧逻辑推进越过快照点 → 结果失效，不回退。
                guard self.sessionGeneration == generationSnapshot,
                      self.recognizedCharCount == pointerSnapshot else { return }

                guard result.available,
                      result.bestIndex >= 0,
                      result.bestIndex < segments.count else { return }

                // 不确定带 [T-δ, T)：本地不推进（极端意译留给 TASK-005 云端补漏）。
                // < T-δ：判为不匹配，不推进。
                guard result.bestScore >= self.semanticThreshold else { return }

                // 命中第 i 段 = 用户已讲到该段末尾。粒度细化：单次最多推进
                // semanticMaxAdvanceSegments 段（cappedIndex），跨多段时靠后续连续
                // 匹配逐段追上；走得更少更保守，不改匹配判定与下方 2/3 防误翻门禁。
                let cappedIndex = min(result.bestIndex, self.semanticMaxAdvanceSegments - 1)
                var advance = pointerSnapshot
                for k in 0...cappedIndex { advance += segments[k].rawLen }
                // 配速锚：语义路径同样夹紧（≥T、2/3、仅向前、过期丢弃均不变）。
                let paceCeiling = self.matchStartOffset + self.spokenScriptUnits(spoken) + self.semanticMaxLeadChars
                let candidate = min(advance, self.sourceText.count, paceCeiling)
                guard candidate > self.recognizedCharCount else { return }

                // 并入同一 2/3 确认门禁；语义跳段不享受 smallStep 直通，必须确认（防误翻）。
                self.recentMatchPositions.append(candidate)
                if self.recentMatchPositions.count > 3 {
                    self.recentMatchPositions.removeFirst()
                }
                let agreementThreshold = 10
                var agreeCount = 0
                for pos in self.recentMatchPositions where abs(pos - candidate) <= agreementThreshold {
                    agreeCount += 1
                }
                if self.recentMatchPositions.count >= 2 && agreeCount >= 2 {
                    self.recognizedCharCount = candidate
                }
            }
        }
    }

    /// 取「用户最近说的话」窗口，代表当前进度。`spoken` 是 SFSpeechRecognizer 单次会话的
    /// 累计转写：优先未结句尾段，否则上一句，否则全句尾部；但**无论是否有句末标点**，
    /// 最终都强制截断到结尾 `semanticWindowMaxChars` 字符，避免中文无标点连续听写时
    /// 窗口随累计转写无限膨胀、语义向量被稀释导致后半段停推（真机验收 r03→r04 修正）。
    private func recentSpokenWindow(_ spoken: String) -> String {
        let trimmed = spoken.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "" }
        let sentenceEnders: Set<Character> = ["。", "！", "？", "；", ".", "!", "?", ";", "\n"]
        var lastSentence = ""
        var current = ""
        for ch in trimmed {
            if sentenceEnders.contains(ch) {
                let s = current.trimmingCharacters(in: .whitespacesAndNewlines)
                if !s.isEmpty { lastSentence = s }
                current = ""
            } else {
                current.append(ch)
            }
        }
        let tail = current.trimmingCharacters(in: .whitespacesAndNewlines)
        let chosen: String
        if tail.count >= 2 {
            chosen = tail
        } else if !lastSentence.isEmpty {
            chosen = lastSentence
        } else {
            chosen = trimmed
        }
        // 关键有界化：窗口必须是「最近一小段」，不能等于整段累计转写。
        if chosen.count > semanticWindowMaxChars {
            return String(chosen.suffix(semanticWindowMaxChars))
        }
        return chosen
    }

    /// 把本会话累计转写投影到与 sourceText 相同单位系（splitTextIntoWords 后空格拼接），
    /// 得到「用户已说出的脚本单位量」，与 recognizedCharCount/matchStartOffset 直接可比。
    private func spokenScriptUnits(_ spoken: String) -> Int {
        splitTextIntoWords(spoken).joined(separator: " ").count
    }

    /// 从指针偏移起、向前至多 maxCount 段。返回每段：可读文本（去掉 CJK 逐字间空格，
    /// 用于语义编码）+ rawLen（该段在 sourceText 中消耗的 Character 数，含分隔符，用于偏移映射）。
    /// 只向前取，天然不向后匹配。
    private func forwardSemanticSegments(fromOffset offset: Int,
                                         maxCount: Int) -> [(text: String, rawLen: Int)] {
        guard offset < sourceText.count else { return [] }
        let chars = Array(sourceText.dropFirst(offset))
        let enders: Set<Character> = ["。", "！", "？", "；", ".", "!", "?", ";"]
        let maxSegmentChars = 60 // 无标点时的兜底切段长度（可读字符计）

        var segments: [(text: String, rawLen: Int)] = []
        var display = ""
        var rawLen = 0
        var readableCount = 0

        func isCJKChar(_ c: Character) -> Bool {
            c.unicodeScalars.first.map { $0.isCJK } ?? false
        }
        func flush() {
            let t = display.trimmingCharacters(in: .whitespacesAndNewlines)
            if !t.isEmpty && rawLen > 0 {
                segments.append((text: t, rawLen: rawLen))
            }
            display = ""
            rawLen = 0
            readableCount = 0
        }

        var idx = 0
        while idx < chars.count && segments.count < maxCount {
            let c = chars[idx]
            rawLen += 1
            if c == " " {
                // splitTextIntoWords 在 CJK 逐字间插了空格：相邻两侧都是 CJK 时丢弃该空格，
                // 还原自然文本；否则（拉丁词边界）保留一个空格。
                let prevCJK = display.last.map { isCJKChar($0) } ?? false
                var nextCJK = false
                if idx + 1 < chars.count { nextCJK = isCJKChar(chars[idx + 1]) }
                if !(prevCJK && nextCJK) && !display.isEmpty && display.last != " " {
                    display.append(" ")
                }
            } else {
                display.append(c)
                readableCount += 1
                if enders.contains(c) {
                    flush()
                } else if readableCount >= maxSegmentChars {
                    flush()
                }
            }
            idx += 1
        }
        if segments.count < maxCount { flush() }
        return segments
    }

    private func charLevelMatch(spoken: String) -> Int {
        let remainingSource = String(sourceText.dropFirst(matchStartOffset))
        // Use Character arrays (not unicodeScalars) so counts match sourceText.count
        let src = Array(remainingSource.lowercased())
        let spk = Array(Self.normalize(spoken))

        var si = 0
        var ri = 0
        var lastGoodOrigIndex = 0

        while si < src.count && ri < spk.count {
            let sc = src[si]
            let rc = spk[ri]

            // Skip non-alphanumeric in source
            if !sc.isLetter && !sc.isNumber {
                si += 1
                continue
            }
            // Skip non-alphanumeric in spoken
            if !rc.isLetter && !rc.isNumber {
                ri += 1
                continue
            }

            if sc == rc {
                si += 1
                ri += 1
                lastGoodOrigIndex = si
            } else {
                // Try to re-sync: look ahead in both strings
                var found = false

                // Skip up to 3 chars in spoken (STT inserted extra chars)
                let maxSkipR = min(3, spk.count - ri - 1)
                if maxSkipR >= 1 {
                    for skipR in 1...maxSkipR {
                        let nextRI = ri + skipR
                        if nextRI < spk.count && spk[nextRI] == sc {
                            ri = nextRI
                            found = true
                            break
                        }
                    }
                }
                if found { continue }

                // Skip up to 3 chars in source (STT missed some chars)
                let maxSkipS = min(3, src.count - si - 1)
                if maxSkipS >= 1 {
                    for skipS in 1...maxSkipS {
                        let nextSI = si + skipS
                        if nextSI < src.count && src[nextSI] == rc {
                            si = nextSI
                            found = true
                            break
                        }
                    }
                }
                if found { continue }

                // No resync found — advance spoken pointer only.
                // Do NOT advance lastGoodOrigIndex; this is a genuine mismatch,
                // not a confirmed match position.
                ri += 1
            }
        }

        return lastGoodOrigIndex
    }

    private static func isAnnotationWord(_ word: String) -> Bool {
        if word.hasPrefix("[") && word.hasSuffix("]") { return true }
        let stripped = word.filter { $0.isLetter || $0.isNumber }
        return stripped.isEmpty
    }

    private func wordLevelMatch(spoken: String) -> Int {
        let remainingSource = String(sourceText.dropFirst(matchStartOffset))
        let sourceWords = remainingSource.split(separator: " ").map { String($0) }
        let spokenWords = spoken.lowercased().split(separator: " ").map { String($0) }

        var si = 0 // source word index
        var ri = 0 // spoken word index
        var matchedCharCount = 0

        while si < sourceWords.count && ri < spokenWords.count {
            // Auto-skip annotation words in source (brackets, emoji)
            if Self.isAnnotationWord(sourceWords[si]) {
                matchedCharCount += sourceWords[si].count
                if si < sourceWords.count - 1 { matchedCharCount += 1 }
                si += 1
                continue
            }

            let srcWord = sourceWords[si].lowercased()
                .filter { $0.isLetter || $0.isNumber }
            let spkWord = spokenWords[ri]
                .filter { $0.isLetter || $0.isNumber }

            if srcWord == spkWord || isFuzzyMatch(srcWord, spkWord) {
                // Count original chars including trailing punctuation
                matchedCharCount += sourceWords[si].count
                si += 1
                ri += 1
                // Add space separator only if there's a following word
                if si < sourceWords.count {
                    matchedCharCount += 1
                }
            } else {
                // Try skipping up to 3 spoken words (STT hallucinated words)
                var foundSpk = false
                let maxSpkSkip = min(3, spokenWords.count - ri - 1)
                for skip in 1...max(1, maxSpkSkip) where skip <= maxSpkSkip {
                    let nextSpk = spokenWords[ri + skip].filter { $0.isLetter || $0.isNumber }
                    if srcWord == nextSpk || isFuzzyMatch(srcWord, nextSpk) {
                        ri += skip
                        foundSpk = true
                        break
                    }
                }
                if foundSpk { continue }

                // Try skipping up to 3 source words (user read fast, STT missed words)
                var foundSrc = false
                let maxSrcSkip = min(3, sourceWords.count - si - 1)
                for skip in 1...max(1, maxSrcSkip) where skip <= maxSrcSkip {
                    let nextSrc = sourceWords[si + skip].lowercased().filter { $0.isLetter || $0.isNumber }
                    if nextSrc == spkWord || isFuzzyMatch(nextSrc, spkWord) {
                        // Add all skipped source words' char counts
                        for s in 0..<skip {
                            matchedCharCount += sourceWords[si + s].count + 1
                        }
                        si += skip
                        foundSrc = true
                        break
                    }
                }
                if foundSrc { continue }

                // Try treating current source word as punctuation-only and skip it
                if srcWord.isEmpty {
                    matchedCharCount += sourceWords[si].count
                    if si < sourceWords.count - 1 { matchedCharCount += 1 }
                    si += 1
                    continue
                }
                // No match, advance spoken
                ri += 1
            }
        }

        // Auto-skip trailing annotation words at end of source
        while si < sourceWords.count && Self.isAnnotationWord(sourceWords[si]) {
            matchedCharCount += sourceWords[si].count
            if si < sourceWords.count - 1 { matchedCharCount += 1 }
            si += 1
        }

        return matchedCharCount
    }

    private func isFuzzyMatch(_ a: String, _ b: String) -> Bool {
        if a.isEmpty || b.isEmpty { return false }
        // Exact match
        if a == b { return true }
        let shorter = min(a.count, b.count)
        // Prefix match — only for words with at least 3 chars to avoid
        // false positives like "or" matching "organization"
        if shorter >= 3 && (a.hasPrefix(b) || b.hasPrefix(a)) { return true }
        // Shared prefix >= 60% of shorter word (min 3 chars shared)
        let shared = zip(a, b).prefix(while: { $0 == $1 }).count
        if shorter >= 3 && shared >= max(3, shorter * 3 / 5) { return true }
        // Edit distance tolerance — stricter for very short words
        let dist = editDistance(a, b)
        if shorter <= 2 { return false } // 2-char words must be exact
        if shorter <= 4 { return dist <= 1 }
        if shorter <= 8 { return dist <= 2 }
        return dist <= max(a.count, b.count) / 3
    }

    private func editDistance(_ a: String, _ b: String) -> Int {
        let a = Array(a), b = Array(b)
        var dp = Array(0...b.count)
        for i in 1...a.count {
            var prev = dp[0]
            dp[0] = i
            for j in 1...b.count {
                let temp = dp[j]
                dp[j] = a[i-1] == b[j-1] ? prev : min(prev, dp[j], dp[j-1]) + 1
                prev = temp
            }
        }
        return dp[b.count]
    }

    private static func normalize(_ text: String) -> String {
        text.lowercased()
            .filter { $0.isLetter || $0.isNumber || $0.isWhitespace }
    }
}
