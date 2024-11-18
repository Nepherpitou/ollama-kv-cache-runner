// Validation functions

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
func ValidateFlashAttentionSupport(ggml GGMLModel, gpus discover.GpuInfoList, flashAttnRequested bool) FlashAttentionSupport {
	isEmbeddingModel := false
	if _, ok := ggml.KV()[fmt.Sprintf("%s.pooling_type", ggml.KV().Architecture())]; ok {
		isEmbeddingModel = true
	}

	// Check model support
	headCountK := ggml.KV().EmbeddingHeadCountK()
	headCountV := ggml.KV().EmbeddingHeadCountV()
	modelSupported := headCountK != 0 && headCountV != 0 && headCountK == headCountV

	if !modelSupported {
		if headCountK == 0 || headCountV == 0 {
			slog.Debug("Model is missing embedding head count for K or V, does not support flash attention")
		} else {
			slog.Debug("Embedding head count K does not equal V, does not support flash attention",
				"K", headCountK,
				"V", headCountV)
		}
	}

	// Check hardware support
	hardwareSupported := true
	for _, g := range gpus {
		// only cuda (compute capability 7+) and metal support flash attention
		if g.Library != "metal" && (g.Library != "cuda" || g.DriverMajor < 7) && g.Library != "rocm" {
			hardwareSupported = false
			break
		}
	}

	// Determine if flash attention should be enabled
	enabled := flashAttnRequested && modelSupported && hardwareSupported && !isEmbeddingModel

	support := FlashAttentionSupport{
		SupportedByModel:    modelSupported,
		SupportedByHardware: hardwareSupported,
		IsEmbeddingModel:    isEmbeddingModel,
		Enabled:             enabled,
	}

	slog.Debug("Flash attention status",
		"supported_by_model", support.SupportedByModel,
		"supported_by_hardware", support.SupportedByHardware,
		"is_embedding_model", support.IsEmbeddingModel,
		"enabled", support.Enabled)

	return support
}

// ValidKVCacheTypes contains all supported KV cache types
var ValidKVCacheTypes = []string{"f32", "f16", "q8_0", "q5_1", "q5_0", "iq4_nl", "q4_1", "q4_0"}

// ValidateKVCacheType checks if the given cache type is valid for the model type
func ValidateKVCacheType(cacheType string, isEmbeddingModel bool) (string, error) {
	if cacheType == "" {
		return "", nil
	}

	if !slices.Contains(ValidKVCacheTypes, cacheType) {
		slog.Warn("invalid cache type, ignoring", "type", cacheType)
		return "", nil
	}

	// For embedding models, only allow f16 and f32
	if isEmbeddingModel && cacheType != "f16" && cacheType != "f32" {
		slog.Warn("only f16 and f32 cache types are supported for embedding models, ignoring",
			"type", cacheType)
		return "", nil
	}

	return cacheType, nil
}

// FlashAttentionSupport contains data about flash attention support
type FlashAttentionSupport struct {
	SupportedByModel    bool
	SupportedByHardware bool
	IsEmbeddingModel    bool
	Enabled             bool
}

// GetServerParams returns the validated and formatted server parameters
func GetServerParams(flashAttn FlashAttentionSupport, kvCacheType string, baseParams []string) []string {
	params := slices.Clone(baseParams)

	if flashAttn.Enabled {
		params = append(params, "--flash-attn")
		slog.Info("Enabling flash attention")

		// Only set KV cache type when flash attention is enabled
		if validatedType, _ := ValidateKVCacheType(kvCacheType, flashAttn.IsEmbeddingModel); validatedType != "" {
			params = append(params, "--kv-cache-type", validatedType)
			slog.Debug("Setting cache type", "type", validatedType)
		}
	} else {
		slog.Info("Flash attention not enabled")
		quantizedCacheTypes := []string{"q8_0", "q5_1", "q5_0", "iq4_nl", "q4_1", "q4_0"}
		if !flashAttn.IsEmbeddingModel && kvCacheType != "" {
			if slices.Contains(quantizedCacheTypes, kvCacheType) {
				slog.Warn("Quantized cache types require flash attention. Falling back to default cache types.")
			}
		}
	}

	return params
}
