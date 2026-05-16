import Foundation
import NaturalLanguage

/// 本地语义匹配引擎。
///
/// 用 Apple `NLContextualEmbedding`（离线、`.simplifiedChinese`）把「用户最近说的话」
/// 与「稿子指针前方的若干候选片段」做语义相似度比对，给出最相近的候选。
/// 这是「意译识别自动翻页」的本地兜底层：旧逐字/逐词匹配拿不准时才由它补判。
///
/// 设计约束（见 knowledge/decisions.md 与 PROJECT-SPEC）：
/// - 纯封装、无 UI、无网络；模型资产缺失/load 失败时安全降级（`isAvailable == false`），
///   调用方据此退回旧逻辑，绝不崩溃。
/// - 只向「前方」候选匹配，不向后（防止说回前文被往回拽）。
/// - 句向量 = token 向量 mean-pool；相似度 = cosine（与 spike 验证口径一致）。
final class SemanticMatcher {

    /// 一次匹配的结果。`available == false` 表示语义层不可用，调用方应退回旧逻辑。
    struct Result {
        /// 命中候选在 `forwardCandidates` 内的下标；无命中为 -1。
        let bestIndex: Int
        /// 命中的 cosine 相似度；无命中/不可用为 .nan。
        let bestScore: Double
        /// 语义层是否可用。
        let available: Bool

        static let unavailable = Result(bestIndex: -1, bestScore: .nan, available: false)
        static let noMatch = Result(bestIndex: -1, bestScore: .nan, available: true)
    }

    private let language: NLLanguage
    private let embedding: NLContextualEmbedding?
    private var loaded = false

    /// 语义层是否就绪可用。`prepare()` 成功后为 true。
    private(set) var isAvailable = false

    init(language: NLLanguage = .simplifiedChinese) {
        self.language = language
        self.embedding = NLContextualEmbedding(language: language)
    }

    /// 一次性加载模型资产。离线安全：资产缺失或 load 失败时返回 false 并保持降级，不抛错、不阻塞。
    /// 建议在跟读启动时于非主线程调用；未就绪期间调用方退回旧匹配。
    @discardableResult
    func prepare() -> Bool {
        guard let embedding else {
            isAvailable = false
            return false
        }
        if !embedding.hasAvailableAssets {
            // 不在此处阻塞下载（提词器需即时可用）。无资产即降级，由调用方走纯旧逻辑。
            isAvailable = false
            return false
        }
        do {
            try embedding.load()
            loaded = true
            isAvailable = true
            return true
        } catch {
            isAvailable = false
            return false
        }
    }

    /// 在「指针前方的候选片段」里找与最近口语窗口语义最相近的一段。
    /// - Parameters:
    ///   - spokenWindow: 用户最近说的话（STT 拼接窗口）。
    ///   - forwardCandidates: 从当前指针起、向前最多 K 段的稿子文本，按先后顺序。
    /// - Returns: 命中下标、分数、是否可用。不可用/空输入/无有效向量时不抛错。
    func bestForwardMatch(spokenWindow: String, forwardCandidates: [String]) -> Result {
        guard isAvailable else { return .unavailable }
        let spoken = spokenWindow.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !spoken.isEmpty, !forwardCandidates.isEmpty,
              let spokenVec = sentenceVector(spoken) else {
            return .noMatch
        }
        var bestIndex = -1
        var bestScore = -Double.infinity
        for (i, candidate) in forwardCandidates.enumerated() {
            let cand = candidate.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !cand.isEmpty, let candVec = sentenceVector(cand) else { continue }
            let score = Self.cosine(spokenVec, candVec)
            guard !score.isNaN else { continue }
            if score > bestScore {
                bestScore = score
                bestIndex = i
            }
        }
        guard bestIndex >= 0 else { return .noMatch }
        return Result(bestIndex: bestIndex, bestScore: bestScore, available: true)
    }

    // MARK: - 句向量

    /// 句向量 = 所有 token 向量的 mean-pool。模型不可用或无 token 时返回 nil。
    func sentenceVector(_ text: String) -> [Double]? {
        guard isAvailable, loaded, let embedding else { return nil }
        guard let result = try? embedding.embeddingResult(for: text, language: language) else {
            return nil
        }
        var vectors: [[Double]] = []
        result.enumerateTokenVectors(in: text.startIndex..<text.endIndex) { vector, _ in
            vectors.append(vector)
            return true
        }
        return Self.meanPool(vectors)
    }

    // MARK: - 纯函数（可独立单测，与 spike 口径一致）

    /// 多个等维向量的逐元素平均。空输入或全空向量返回 nil。
    static func meanPool(_ vectors: [[Double]]) -> [Double]? {
        let nonEmpty = vectors.filter { !$0.isEmpty }
        guard let first = nonEmpty.first else { return nil }
        var acc = [Double](repeating: 0, count: first.count)
        var count = 0
        for v in nonEmpty {
            let n = min(acc.count, v.count)
            for i in 0..<n { acc[i] += v[i] }
            count += 1
        }
        guard count > 0 else { return nil }
        return acc.map { $0 / Double(count) }
    }

    /// 余弦相似度。维度不匹配、空向量或零范数返回 .nan。
    static func cosine(_ a: [Double], _ b: [Double]) -> Double {
        guard a.count == b.count, !a.isEmpty else { return .nan }
        var dot = 0.0, na = 0.0, nb = 0.0
        for i in 0..<a.count {
            dot += a[i] * b[i]
            na += a[i] * a[i]
            nb += b[i] * b[i]
        }
        guard na > 0, nb > 0 else { return .nan }
        return dot / (sqrt(na) * sqrt(nb))
    }
}
