// validation.go

package llm

import (
	"fmt"
	"log/slog"
	"slices"

	"github.com/ollama/ollama/discover"
)

// Interface for GGML functionality needed by validation
type GGMLModel interface {
	KV() KV
}

// ValidateFlashAttentionSupport checks if flash attention is supported by the model and hardware
func ValidateFlashAttentionSupport(ggml GGMLModel, gpus discover.GpuInfoList, flashAttnRequested bool) bool {
	if !gpus.SupportsFlashAttention() {
		return false
	}

	return supportsFlashAttention(ggml) && flashAttnRequested
}

// supportsFlashAttention checks if the model supports flash attention
func supportsFlashAttention(ggml GGMLModel) bool {
	// Check if it's an embedding model - embedding models don't support flash attention
	if _, ok := ggml.KV()[fmt.Sprintf("%s.pooling_type", ggml.KV().Architecture())]; ok {
		return false
	}

	// Check head counts match and are non-zero
	headCountK := ggml.KV().EmbeddingHeadCountK()
	headCountV := ggml.KV().EmbeddingHeadCountV()
	return headCountK != 0 && headCountV != 0 && headCountK == headCountV
}

// ValidKVCacheTypes contains all supported KV cache types
var ValidKVCacheTypes = []string{"f32", "f16", "q8_0", "q5_1", "q5_0", "iq4_nl", "q4_1", "q4_0"}

// ValidateKVCacheType checks if the given cache type is valid for the model type
func ValidateKVCacheType(cacheType string, isEmbedding bool) (string, error) {
	if cacheType == "" {
		return "", nil
	}

	if !slices.Contains(ValidKVCacheTypes, cacheType) {
		slog.Warn("invalid cache type, defaulting to f16", "type", cacheType)
		return "f16", nil
	}

	// For embedding models, only allow f16 and f32
	if isEmbedding && cacheType != "f16" && cacheType != "f32" {
		slog.Warn("only f16 and f32 cache types are supported for embedding models, defaulting to f16",
			"type", cacheType)
		return "f16", nil
	}

	return cacheType, nil
}

// GetServerParams returns the validated and formatted server parameters
func GetServerParams(ggml GGMLModel, gpus discover.GpuInfoList, flashAttnRequested bool, kvCacheType string, baseParams []string) []string {
	params := slices.Clone(baseParams)

	flashAttnEnabled := ValidateFlashAttentionSupport(ggml, gpus, flashAttnRequested)
	isEmbeddingModel := false
	if _, ok := ggml.KV()[fmt.Sprintf("%s.pooling_type", ggml.KV().Architecture())]; ok {
		isEmbeddingModel = true
	}

	if flashAttnEnabled {
		params = append(params, "--flash-attn")
		slog.Info("Enabling flash attention")

		// Only set KV cache type when flash attention is enabled
		if validatedType, _ := ValidateKVCacheType(kvCacheType, isEmbeddingModel); validatedType != "" {
			params = append(params, "--kv-cache-type", validatedType)
			slog.Debug("Setting cache type", "type", validatedType)
		}
	} else {
		slog.Info("Flash attention not enabled")
		if !isEmbeddingModel && kvCacheType != "" {
			quantizedTypes := []string{"q8_0", "q5_1", "q5_0", "iq4_nl", "q4_1", "q4_0"}
			if slices.Contains(quantizedTypes, kvCacheType) {
				slog.Warn("Quantized cache types require flash attention. Using default cache type.")
			}
		}
	}

	return params
}
