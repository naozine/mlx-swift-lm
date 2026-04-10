//
//  Gemma4.swift
//  mlx-swift-lm
//
//  Based on SharpAI/mlx-swift-lm Gemma4 implementation
//  and https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gemma3_text.py

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct Gemma4Configuration: Codable {
    let modelType: String
    let hiddenSize: Int
    let hiddenLayers: Int
    let intermediateSize: Int
    let attentionHeads: Int
    let headDim: Int
    let rmsNormEps: Float
    let vocabularySize: Int
    let kvHeads: Int
    let ropeTheta: Float
    let ropeLocalBaseFreq: Float
    let ropeTraditional: Bool
    let queryPreAttnScalar: Float?
    let slidingWindow: Int
    let slidingWindowPattern: Int
    let maxPositionEmbeddings: Int
    let ropeScaling: [String: StringOrNumber]?
    let globalHeadDim: Int
    let numKvSharedLayers: Int
    let useDoubleWideMlp: Bool

    public let numExperts: Int?
    public let topKExperts: Int?
    public let moeIntermediateSize: Int?
    public let numGlobalKeyValueHeads: Int
    public let tieWordEmbeddings: Bool

    let globalRopePartialFactor: Float
    let finalLogitSoftcapping: Float
    let hiddenSizePerLayerInput: Int
    let vocabSizePerLayerInput: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case ropeTheta = "rope_theta"
        case ropeLocalBaseFreq = "rope_local_base_freq"
        case ropeTraditional = "rope_traditional"
        case queryPreAttnScalar = "query_pre_attn_scalar"
        case slidingWindow = "sliding_window"
        case slidingWindowPattern = "sliding_window_pattern"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeScaling = "rope_scaling"
        case globalHeadDim = "global_head_dim"
        case numKvSharedLayers = "num_kv_shared_layers"
        case useDoubleWideMlp = "use_double_wide_mlp"
        case tieWordEmbeddings = "tie_word_embeddings"
        case numGlobalKeyValueHeads = "num_global_key_value_heads"
        case numExperts = "num_experts"
        case topKExperts = "top_k_experts"
        case moeIntermediateSize = "moe_intermediate_size"
        case hiddenSizePerLayerInput = "hidden_size_per_layer_input"
        case vocabSizePerLayerInput = "vocab_size_per_layer_input"
        case finalLogitSoftcapping = "final_logit_softcapping"
    }

    enum VLMCodingKeys: String, CodingKey {
        case textConfig = "text_config"
    }

    public init(from decoder: Decoder) throws {
        let nestedContainer = try decoder.container(keyedBy: VLMCodingKeys.self)

        let container =
            if nestedContainer.contains(.textConfig) {
                try nestedContainer.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig)
            } else {
                try decoder.container(keyedBy: CodingKeys.self)
            }

        modelType = try container.decode(String.self, forKey: .modelType)
        tieWordEmbeddings = try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        numExperts = try container.decodeIfPresent(Int.self, forKey: .numExperts)
        topKExperts = try container.decodeIfPresent(Int.self, forKey: .topKExperts)
        moeIntermediateSize = try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize)
        numGlobalKeyValueHeads = try container.decodeIfPresent(Int.self, forKey: .numGlobalKeyValueHeads)
            ?? (try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 1)
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1152
        hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 26
        intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 6912
        attentionHeads = try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 4
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 256
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1.0e-6
        vocabularySize = try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 262144
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 1
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000.0
        ropeLocalBaseFreq = try container.decodeIfPresent(Float.self, forKey: .ropeLocalBaseFreq) ?? 10_000.0
        ropeTraditional = try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? true
        queryPreAttnScalar = try container.decodeIfPresent(Float.self, forKey: .queryPreAttnScalar)
        finalLogitSoftcapping = try container.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping) ?? 0.0
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 512
        slidingWindowPattern = try container.decodeIfPresent(Int.self, forKey: .slidingWindowPattern)
            ?? (hiddenLayers == 35 ? 5 : 6)
        maxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32768
        ropeScaling = try container.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
        globalHeadDim = try container.decodeIfPresent(Int.self, forKey: .globalHeadDim) ?? 512
        numKvSharedLayers = try container.decodeIfPresent(Int.self, forKey: .numKvSharedLayers) ?? 0
        useDoubleWideMlp = try container.decodeIfPresent(Bool.self, forKey: .useDoubleWideMlp) ?? false
        hiddenSizePerLayerInput = try container.decodeIfPresent(Int.self, forKey: .hiddenSizePerLayerInput) ?? 0
        vocabSizePerLayerInput = try container.decodeIfPresent(Int.self, forKey: .vocabSizePerLayerInput) ?? 0

        // Parse partial_rotary_factor from rope_parameters
        struct AC: CodingKey {
            var stringValue: String; init?(stringValue s: String) { stringValue = s }
            var intValue: Int? { nil }; init?(intValue _: Int) { nil }
        }
        if let nestedContainer = try? decoder.container(keyedBy: AC.self),
           let ropeParamsContainer = try? nestedContainer.nestedContainer(keyedBy: AC.self, forKey: AC(stringValue: "rope_parameters")!),
           let fullAttnContainer = try? ropeParamsContainer.nestedContainer(keyedBy: AC.self, forKey: AC(stringValue: "full_attention")!),
           let prf = try? fullAttnContainer.decode(Float.self, forKey: AC(stringValue: "partial_rotary_factor")!) {
            self.globalRopePartialFactor = prf
        } else {
            self.globalRopePartialFactor = 0.25
        }
    }
}

// MARK: - RMSNorm without learnable scale

public class Gemma4RMSNormNoScale: Module {
    let eps: Float

    public init(eps: Float = 1e-6) {
        self.eps = eps
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xFloat32 = x.asType(.float32)
        let meanSq = MLX.mean(MLX.square(xFloat32), axes: [-1], keepDims: true)
        let inverseNorm = MLX.rsqrt(meanSq + eps)
        return (xFloat32 * inverseNorm).asType(x.dtype)
    }
}

// MARK: - Proportional RoPE for global attention

public class Gemma4ProportionalRoPE: Module, OffsetLayer {
    let dims: Int
    let traditional: Bool
    let rotatedDims: Int
    private var _computedFreqs: MLXArray?

    public init(dims: Int, traditional: Bool = false, base: Float = 10000.0, partialRotaryFactor: Float = 1.0) {
        self.dims = dims
        self.traditional = traditional

        let ropeAngles = Int(partialRotaryFactor * Float(dims) / 2.0)
        self.rotatedDims = 2 * ropeAngles

        if rotatedDims > 0 {
            let exponents = MLXArray(stride(from: Float(0), to: Float(rotatedDims), by: 2)) / Float(dims)
            self._computedFreqs = MLXArray(base) ** exponents
        } else {
            self._computedFreqs = nil
        }

        super.init()
        self.freeze()
    }

    public func callAsFunction(_ x: MLXArray, offset: Int) -> MLXArray {
        guard rotatedDims > 0, let freqs = _computedFreqs else { return x }

        let head = x[0..., 0..., 0..., 0..<dims]
        let half = dims / 2

        let left = head[0..., 0..., 0..., 0..<half]
        let right = head[0..., 0..., 0..., half...]

        let rotHalf = rotatedDims / 2

        let leftRot = left[0..., 0..., 0..., 0..<rotHalf]
        let rightRot = right[0..., 0..., 0..., 0..<rotHalf]
        let rotated = concatenated([leftRot, rightRot], axis: -1)

        let rotatedResult = MLXFast.RoPE(
            rotated, dimensions: rotatedDims, traditional: traditional,
            base: nil, scale: 1.0, offset: offset, freqs: freqs)

        let leftPassthru = left[0..., 0..., 0..., rotHalf...]
        let rightPassthru = right[0..., 0..., 0..., rotHalf...]

        let newLeft = concatenated([rotatedResult[0..., 0..., 0..., 0..<rotHalf], leftPassthru], axis: -1)
        let newRight = concatenated([rotatedResult[0..., 0..., 0..., rotHalf...], rightPassthru], axis: -1)

        return concatenated([newLeft, newRight], axis: -1)
    }
}

// MARK: - Attention

class Gemma4Attention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let repeats: Int
    let headDim: Int
    let layerIdx: Int
    let scale: Float
    let isSliding: Bool
    let slidingWindow: Int
    let slidingWindowPattern: Int

    @ModuleInfo(key: "q_proj") var queryProj: Linear
    @ModuleInfo(key: "k_proj") var keyProj: Linear
    @ModuleInfo(key: "v_proj") var valueProj: Linear
    @ModuleInfo(key: "o_proj") var outputProj: Linear

    @ModuleInfo(key: "q_norm") var queryNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: RMSNorm
    @ModuleInfo(key: "v_norm") var valueNorm: Gemma4RMSNormNoScale

    @ModuleInfo var rope: OffsetLayer

    init(_ config: Gemma4Configuration, layerIdx: Int) {
        let dim = config.hiddenSize
        self.layerIdx = layerIdx
        self.slidingWindow = config.slidingWindow
        self.slidingWindowPattern = config.slidingWindowPattern
        self.isSliding = (layerIdx + 1) % config.slidingWindowPattern != 0

        self.nHeads = config.attentionHeads
        self.nKVHeads = self.isSliding ? config.kvHeads : config.numGlobalKeyValueHeads
        self.repeats = nHeads / (nKVHeads > 0 ? nKVHeads : 1)
        self.headDim = self.isSliding ? config.headDim : config.globalHeadDim

        self.scale = 1.0

        self._queryProj.wrappedValue = Linear(dim, nHeads * self.headDim, bias: false)
        self._keyProj.wrappedValue = Linear(dim, nKVHeads * self.headDim, bias: false)
        self._valueProj.wrappedValue = Linear(dim, nKVHeads * self.headDim, bias: false)
        self._outputProj.wrappedValue = Linear(nHeads * self.headDim, dim, bias: false)

        self._queryNorm.wrappedValue = RMSNorm(dimensions: self.headDim, eps: config.rmsNormEps)
        self._keyNorm.wrappedValue = RMSNorm(dimensions: self.headDim, eps: config.rmsNormEps)
        self._valueNorm.wrappedValue = Gemma4RMSNormNoScale(eps: config.rmsNormEps)

        if isSliding {
            self.rope = RoPE(
                dimensions: headDim, traditional: false,
                base: config.ropeLocalBaseFreq, scale: 1.0)
        } else {
            self.rope = Gemma4ProportionalRoPE(
                dims: self.headDim,
                traditional: false,
                base: 1000000.0,
                partialRotaryFactor: config.globalRopePartialFactor
            )
        }

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache? = nil
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var queries = queryProj(x)
        var keys = keyProj(x)
        var values = valueProj(x)

        queries = queries.reshaped(B, L, nHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)

        queries = queryNorm(queries)
        keys = keyNorm(keys)
        values = valueNorm(values)

        let LCache = cache?.offset ?? 0
        keys = rope(keys, offset: LCache)
        queries = rope(queries, offset: LCache)

        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values,
            cache: cache, scale: scale, mask: mask)

        return outputProj(
            output.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        )
    }
}

// MARK: - MLP

public class Gemma4MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear

    public init(dimensions: Int, hiddenDimensions: Int) {
        self._gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        self._upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

// MARK: - MoE Router

class Gemma4Router: Module {
    @ModuleInfo(key: "proj") var proj: Linear
    @ModuleInfo(key: "scale") var scale: MLXArray
    @ModuleInfo(key: "per_expert_scale") var perExpertScale: MLXArray

    let eps: Float
    let scalarRootSize: Float

    public init(dimensions: Int, numExperts: Int, eps: Float) {
        self.eps = eps
        self.scalarRootSize = 1.0 / sqrt(Float(dimensions))
        self._proj.wrappedValue = Linear(dimensions, numExperts, bias: false)
        self._scale.wrappedValue = MLXArray.ones([dimensions])
        self._perExpertScale.wrappedValue = MLXArray.ones([numExperts])
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, topK: Int) -> (MLXArray, MLXArray) {
        let xF32 = x.asType(.float32)
        let meanX2 = MLX.mean(MLX.square(xF32), axes: [-1], keepDims: true)
        let xNormed = (x * MLX.rsqrt(meanX2 + MLXArray(eps))).asType(x.dtype)

        let scaled = xNormed * MLXArray(scalarRootSize) * scale

        let expertScores = proj(scaled)
        let routerProbs = MLX.softmax(expertScores, axis: -1)

        let negScores = MLX.negative(expertScores)
        let allInds = MLX.argPartition(negScores, kth: topK - 1, axis: -1)
        let topKInds = allInds[0..., 0..., 0..<topK]

        var topKWeights = MLX.takeAlong(routerProbs, topKInds, axis: -1)
        topKWeights = topKWeights / topKWeights.sum(axis: -1, keepDims: true)
        topKWeights = topKWeights * perExpertScale[topKInds]

        return (topKWeights, topKInds)
    }
}

// MARK: - Sparse MoE Block

class Gemma4SparseMoeBlock: Module {
    let topK: Int

    @ModuleInfo(key: "switch_glu") var switchGLU: SwitchGLU
    @ModuleInfo(key: "router") var router: Gemma4Router

    init(dimensions: Int, numExperts: Int, topK: Int, moeIntermediateSize: Int) {
        self.topK = topK
        self._router.wrappedValue = Gemma4Router(dimensions: dimensions, numExperts: numExperts, eps: 1e-6)
        self._switchGLU.wrappedValue = SwitchGLU(
            inputDims: dimensions,
            hiddenDims: moeIntermediateSize,
            numExperts: numExperts,
            activation: geluApproximate
        )
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, routerInput: MLXArray) -> MLXArray {
        let (scores, inds) = router(routerInput, topK: topK)
        let y = switchGLU(x, inds)

        let B = y.dim(0)
        let T = y.dim(1)

        let yMasked = y * scores[0..., 0..., 0..., .newAxis]
        let yMerged = yMasked.sum(axis: 2)

        return yMerged.reshaped([B, T, -1])
    }
}

// MARK: - Transformer Block

class Gemma4TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: Gemma4Attention
    @ModuleInfo var mlp: Gemma4MLP
    @ModuleInfo(key: "experts") var expertsBlock: Gemma4SparseMoeBlock?

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: RMSNorm

    @ModuleInfo(key: "post_feedforward_layernorm_1") var postFeedforwardLayerNorm1: RMSNorm?
    @ModuleInfo(key: "pre_feedforward_layernorm_2") var preFeedforwardLayerNorm2: RMSNorm?
    @ModuleInfo(key: "post_feedforward_layernorm_2") var postFeedforwardLayerNorm2: RMSNorm?

    @ModuleInfo(key: "per_layer_input_gate") var perLayerInputGate: Linear?
    @ModuleInfo(key: "per_layer_projection") var perLayerProjectionLayer: Linear?
    @ModuleInfo(key: "post_per_layer_input_norm") var postPerLayerInputNorm: RMSNorm?

    @ModuleInfo(key: "layer_scalar") var layerScalar: MLXArray

    let hiddenSize: Int
    let layerIdx: Int
    let isMoe: Bool
    let hasPerLayerInput: Bool

    init(_ config: Gemma4Configuration, layerIdx: Int) {
        self.hiddenSize = config.hiddenSize
        self.layerIdx = layerIdx
        self.hasPerLayerInput = config.hiddenSizePerLayerInput > 0

        self._selfAttention.wrappedValue = Gemma4Attention(config, layerIdx: layerIdx)

        let mlpSize: Int
        if config.useDoubleWideMlp && layerIdx >= (config.hiddenLayers - config.numKvSharedLayers) {
            mlpSize = config.intermediateSize * 2
        } else {
            mlpSize = config.intermediateSize
        }
        self.mlp = Gemma4MLP(dimensions: config.hiddenSize, hiddenDimensions: mlpSize)

        self.isMoe = config.numExperts != nil && config.numExperts! > 0

        if self.isMoe {
            let numExperts = config.numExperts!
            self._expertsBlock.wrappedValue = Gemma4SparseMoeBlock(
                dimensions: config.hiddenSize,
                numExperts: numExperts,
                topK: config.topKExperts ?? 1,
                moeIntermediateSize: config.moeIntermediateSize ?? config.intermediateSize
            )
        }

        if self.isMoe {
            self._postFeedforwardLayerNorm1.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._preFeedforwardLayerNorm2.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._postFeedforwardLayerNorm2.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        if hasPerLayerInput {
            self._perLayerInputGate.wrappedValue = Linear(
                config.hiddenSize, config.hiddenSizePerLayerInput, bias: false)
            self._perLayerProjectionLayer.wrappedValue = Linear(
                config.hiddenSizePerLayerInput, config.hiddenSize, bias: false)
            self._postPerLayerInputNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._preFeedforwardLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._layerScalar.wrappedValue = MLXArray.ones([1])

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache? = nil,
        perLayerInput: MLXArray? = nil
    ) -> MLXArray {
        let inputNorm = inputLayerNorm(x)
        let r = selfAttention(inputNorm, mask: mask, cache: cache)
        let attnNorm = postAttentionLayerNorm(r)

        var residualUpdates: MLXArray

        if isMoe {
            let preMLPNorm = preFeedforwardLayerNorm(x + attnNorm)
            let denseOut = mlp(preMLPNorm)
            let densePostNorm1 = postFeedforwardLayerNorm1!(denseOut)

            let routerInput = x + attnNorm
            let sparsePreNorm = preFeedforwardLayerNorm2!(routerInput)
            let sparseOut = expertsBlock!(sparsePreNorm, routerInput: routerInput)
            let sparsePostNorm2 = postFeedforwardLayerNorm2!(sparseOut)

            let combined = densePostNorm1 + sparsePostNorm2
            let postMLPNorm = postFeedforwardLayerNorm(combined)
            residualUpdates = attnNorm + postMLPNorm
        } else {
            let preMLPNorm = preFeedforwardLayerNorm(x + attnNorm)
            let r2 = mlp(preMLPNorm)
            let postMLPNorm = postFeedforwardLayerNorm(r2)
            residualUpdates = attnNorm + postMLPNorm
        }

        if hasPerLayerInput,
           let pli = perLayerInput,
           let gate = perLayerInputGate,
           let proj = perLayerProjectionLayer,
           let norm = postPerLayerInputNorm
        {
            var gated = gate(x + residualUpdates)
            gated = geluApproximate(gated)
            gated = gated * pli
            gated = proj(gated)
            gated = norm(gated)
            residualUpdates = residualUpdates + gated
        }

        return (x + residualUpdates) * layerScalar
    }
}

// MARK: - Model Internal

public class Gemma4ModelInternal: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [Gemma4TransformerBlock]
    @ModuleInfo var norm: RMSNorm

    @ModuleInfo(key: "embed_tokens_per_layer") var embedTokensPerLayer: Embedding?
    @ModuleInfo(key: "per_layer_model_projection") var perLayerModelProjection: Linear?
    @ModuleInfo(key: "per_layer_projection_norm") var perLayerProjectionNorm: RMSNorm?

    let config: Gemma4Configuration

    init(_ config: Gemma4Configuration) {
        self.config = config

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )

        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map { layerIdx in
            Gemma4TransformerBlock(config, layerIdx: layerIdx)
        }

        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        if config.hiddenSizePerLayerInput > 0 {
            self._embedTokensPerLayer.wrappedValue = Embedding(
                embeddingCount: config.vocabSizePerLayerInput,
                dimensions: config.hiddenLayers * config.hiddenSizePerLayerInput
            )
            self._perLayerModelProjection.wrappedValue = Linear(
                config.hiddenSize,
                config.hiddenLayers * config.hiddenSizePerLayerInput,
                bias: false
            )
            self._perLayerProjectionNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSizePerLayerInput, eps: config.rmsNormEps)
        }

        super.init()
    }

    func callAsFunction(
        _ inputs: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: [KVCache?]? = nil
    ) -> MLXArray {
        var h = embedTokens(inputs)
        // Gemma 4 embedding scaling: h *= sqrt(hidden_size)
        h = h * MLXArray(Float(config.hiddenSize).squareRoot())

        var layerCache = cache
        if layerCache == nil {
            layerCache = Array(repeating: nil as KVCache?, count: layers.count)
        }

        let globalMask = createAttentionMask(h: h, cache: cache?[config.slidingWindowPattern - 1])
        let slidingWindowMask: MLXFast.ScaledDotProductAttentionMaskMode =
            config.slidingWindowPattern > 1
            ? createAttentionMask(h: h, cache: cache?[0], windowSize: config.slidingWindow)
            : .none

        // Per-layer conditioning
        var perLayerInputs: MLXArray? = nil
        if config.hiddenSizePerLayerInput > 0,
           let embedPerLayer = embedTokensPerLayer,
           let modelProj = perLayerModelProjection,
           let projNorm = perLayerProjectionNorm
        {
            let B = inputs.dim(0)
            let L = inputs.dim(1)
            let nL = config.hiddenLayers
            let D = config.hiddenSizePerLayerInput

            let tokenScale = MLXArray(sqrt(Float(D))).asType(h.dtype)
            let tokenEmbeds = (embedPerLayer(inputs) * tokenScale).reshaped(B, L, nL, D)

            let projScale = MLXArray(1.0 / sqrt(Float(config.hiddenSize))).asType(h.dtype)
            let modelProjected = (modelProj(h) * projScale).reshaped(B, L, nL, D)
            let modelProjectedNormed = projNorm(modelProjected)

            let combineScale = MLXArray(Float(1.0 / 2.0.squareRoot())).asType(h.dtype)
            perLayerInputs = (tokenEmbeds + modelProjectedNormed) * combineScale
        }

        for (i, layer) in layers.enumerated() {
            let isGlobal = (i % config.slidingWindowPattern == config.slidingWindowPattern - 1)
            let layerMask = isGlobal ? globalMask : slidingWindowMask
            let pli = perLayerInputs.map { $0[0..., 0..., i, 0...] }
            h = layer(h, mask: layerMask, cache: layerCache?[i], perLayerInput: pli)
        }
        return norm(h)
    }
}

// MARK: - Top-level Model

public class Gemma4Model: Module, LLMModel {

    @ModuleInfo public var model: Gemma4ModelInternal
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public let config: Gemma4Configuration
    public var vocabularySize: Int { config.vocabularySize }

    public init(_ config: Gemma4Configuration) {
        self.config = config
        self.model = Gemma4ModelInternal(config)
        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var out = model(inputs, mask: nil, cache: cache)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        if config.finalLogitSoftcapping > 0 {
            let cap = MLXArray(config.finalLogitSoftcapping)
            out = MLX.tanh(out / cap) * cap
        }
        return out
    }

    public func sanitize(weights: [String: MLXArray], metadata: [String: String]) -> [String: MLXArray] {
        var processedWeights = weights

        let unflattened = ModuleParameters.unflattened(weights)
        if let lm = unflattened["language_model"] {
            processedWeights = Dictionary(uniqueKeysWithValues: lm.flattened())
        }

        let expectedVocab = config.vocabularySize
        let keysToCheck = [
            "model.embed_tokens.weight", "model.embed_tokens.scales", "model.embed_tokens.biases",
            "lm_head.weight", "lm_head.scales", "lm_head.biases",
        ]
        for key in keysToCheck {
            if let tensor = processedWeights[key], tensor.dim(0) > expectedVocab {
                processedWeights[key] = tensor[0 ..< expectedVocab]
            }
        }

        var finalWeights = [String: MLXArray]()
        for (k, v) in processedWeights {
            if k.contains("self_attn.rotary_emb") || k.contains("input_max") || k.contains("input_min")
                || k.contains("output_max") || k.contains("output_min") {
                continue
            }
            if k.hasSuffix(".experts.gate_up_proj.weight") {
                let base = k.replacingOccurrences(of: ".experts.gate_up_proj.weight", with: ".experts.switch_glu")
                let parts = MLX.split(v, parts: 2, axis: -2)
                finalWeights["\(base).gate_proj.weight"] = parts[0]
                finalWeights["\(base).up_proj.weight"] = parts[1]
                continue
            }
            if k.hasSuffix(".experts.down_proj.weight") {
                let base = k.replacingOccurrences(of: ".experts.down_proj.weight", with: ".experts.switch_glu.down_proj.weight")
                finalWeights[base] = v
                continue
            }
            let newK = k.replacingOccurrences(of: ".router.", with: ".experts.router.")
            finalWeights[newK] = v
        }

        if let normWeight = weights["language_model.model.per_layer_projection_norm.weight"] {
            finalWeights["model.per_layer_projection_norm.weight"] = normWeight
        } else if let normWeight = weights["model.per_layer_projection_norm.weight"] {
            finalWeights["model.per_layer_projection_norm.weight"] = normWeight
        }

        // Dequantize router proj weights (may be quantized differently from other layers)
        for i in 0..<config.hiddenLayers {
            let wKey = "model.layers.\(i).experts.router.proj.weight"
            let sKey = "model.layers.\(i).experts.router.proj.scales"
            let bKey = "model.layers.\(i).experts.router.proj.biases"
            if let packedW = finalWeights[wKey],
               let scales = finalWeights[sKey],
               let biases = finalWeights[bKey] {
                let bits = 32 * packedW.shape.last! / (scales.shape.last! * 64)
                finalWeights[wKey] = MLX.dequantized(
                    packedW, scales: scales, biases: biases, groupSize: 64, bits: bits)
                finalWeights.removeValue(forKey: sKey)
                finalWeights.removeValue(forKey: bKey)
            }
        }

        // Gemma 4 shares k_proj weights with v_proj
        for i in 0..<config.hiddenLayers {
            let kWeightKey = "model.layers.\(i).self_attn.k_proj.weight"
            let vWeightKey = "model.layers.\(i).self_attn.v_proj.weight"
            if finalWeights[kWeightKey] != nil && finalWeights[vWeightKey] == nil {
                finalWeights[vWeightKey] = finalWeights[kWeightKey]
                for suffix in ["scales", "biases"] {
                    let kKey = "model.layers.\(i).self_attn.k_proj.\(suffix)"
                    let vKey = "model.layers.\(i).self_attn.v_proj.\(suffix)"
                    if finalWeights[kKey] != nil {
                        finalWeights[vKey] = finalWeights[kKey]
                    }
                }
            }
        }

        return finalWeights
    }

    public func newCache(parameters: GenerateParameters? = nil) -> [KVCache] {
        var caches = [KVCache]()
        let slidingWindow = config.slidingWindow
        let slidingWindowPattern = config.slidingWindowPattern

        for i in 0 ..< config.hiddenLayers {
            let isGlobalLayer = (i % slidingWindowPattern == slidingWindowPattern - 1)

            if isGlobalLayer {
                let cache = StandardKVCache()
                cache.step = 1024
                caches.append(cache)
            } else {
                caches.append(
                    RotatingKVCache(maxSize: slidingWindow, keep: 0)
                )
            }
        }

        return caches
    }
}

extension Gemma4Model: LoRAModel {
    public var loraLayers: [Module] {
        model.layers.map { $0 as Module }
    }
}
