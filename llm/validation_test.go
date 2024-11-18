package llm

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/ollama/ollama/discover"
)

// testGGML implements GGMLModel for testing
type testGGML struct {
	kv KV
}

func (g *testGGML) KV() KV {
	return g.kv
}

func TestValidateKVCacheType(t *testing.T) {
	tests := []struct {
		name             string
		cacheType        string
		isEmbeddingModel bool
		want             string
		wantErr          bool
	}{
		{
			name:             "valid type for normal model",
			cacheType:        "q8_0",
			isEmbeddingModel: false,
			want:             "q8_0",
			wantErr:          false,
		},
		{
			name:             "invalid type",
			cacheType:        "invalid",
			isEmbeddingModel: false,
			want:             "",
			wantErr:          false,
		},
		{
			name:             "quantized type for embedding model",
			cacheType:        "q8_0",
			isEmbeddingModel: true,
			want:             "",
			wantErr:          false,
		},
		{
			name:             "valid type for embedding model",
			cacheType:        "f16",
			isEmbeddingModel: true,
			want:             "f16",
			wantErr:          false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ValidateKVCacheType(tt.cacheType, tt.isEmbeddingModel)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestValidateFlashAttentionSupport(t *testing.T) {
	tests := []struct {
		name               string
		kvData             map[string]any
		gpus               discover.GpuInfoList
		flashAttnRequested bool
		want               FlashAttentionSupport
	}{
		{
			name: "supported model and hardware",
			kvData: map[string]any{
				"general.architecture":         "llama",
				"llama.attention.key_length":   uint32(32),
				"llama.attention.value_length": uint32(32),
			},
			gpus: discover.GpuInfoList{
				{Library: "cuda", DriverMajor: 8},
			},
			flashAttnRequested: true,
			want: FlashAttentionSupport{
				SupportedByModel:    true,
				SupportedByHardware: true,
				IsEmbeddingModel:    false,
				Enabled:             true,
			},
		},
		{
			name: "embedding model",
			kvData: map[string]any{
				"general.architecture":        "bert",
				"bert.attention.key_length":   uint32(32),
				"bert.attention.value_length": uint32(32),
				"bert.pooling_type":           "mean",
			},
			gpus: discover.GpuInfoList{
				{Library: "cuda", DriverMajor: 8},
			},
			flashAttnRequested: true,
			want: FlashAttentionSupport{
				SupportedByModel:    true,
				SupportedByHardware: true,
				IsEmbeddingModel:    true,
				Enabled:             false,
			},
		},
		{
			name: "unsupported hardware",
			kvData: map[string]any{
				"general.architecture":         "llama",
				"llama.attention.key_length":   uint32(32),
				"llama.attention.value_length": uint32(32),
			},
			gpus: discover.GpuInfoList{
				{Library: "cuda", DriverMajor: 6},
			},
			flashAttnRequested: true,
			want: FlashAttentionSupport{
				SupportedByModel:    true,
				SupportedByHardware: false,
				IsEmbeddingModel:    false,
				Enabled:             false,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ggml := &testGGML{kv: tt.kvData}
			got := ValidateFlashAttentionSupport(ggml, tt.gpus, tt.flashAttnRequested)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestGetServerParams(t *testing.T) {
	tests := []struct {
		name        string
		flashAttn   FlashAttentionSupport
		kvCacheType string
		baseParams  []string
		want        []string
	}{
		{
			name: "flash attention enabled with valid cache type",
			flashAttn: FlashAttentionSupport{
				Enabled:          true,
				IsEmbeddingModel: false,
			},
			kvCacheType: "q8_0",
			baseParams:  []string{"--model", "test"},
			want:        []string{"--model", "test", "--flash-attn", "--kv-cache-type", "q8_0"},
		},
		{
			name: "flash attention disabled",
			flashAttn: FlashAttentionSupport{
				Enabled:          false,
				IsEmbeddingModel: false,
			},
			kvCacheType: "q8_0",
			baseParams:  []string{"--model", "test"},
			want:        []string{"--model", "test"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := GetServerParams(tt.flashAttn, tt.kvCacheType, tt.baseParams)
			assert.Equal(t, tt.want, got)
		})
	}
}
