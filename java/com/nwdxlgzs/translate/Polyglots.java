package com.nwdxlgzs.translate;

import android.annotation.SuppressLint;

import org.json.JSONException;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.PriorityQueue;

import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;

public class Polyglots {
    public final M2M100TokenizerFast tokenizer;
    public final M2MEncoder encoder;
    public final M2MDecoder decoder;
    public final M2MConfig config;

    public Polyglots(OrtEnvironment ortEnv, String modelDir) throws JSONException, IOException {
        tokenizer = M2M100TokenizerFast.fromPretrained(modelDir);
        encoder = new M2MEncoder(ortEnv, new File(modelDir, "encoder_model.onnx").getAbsolutePath());
        decoder = new M2MDecoder(ortEnv, new File(modelDir, "decoder_model_merged.onnx").getAbsolutePath());
        config = new M2MConfig(new File(modelDir, "config.json").getAbsolutePath(), new File(modelDir, "generation_config.json").getAbsolutePath());
    }

    private void addPastKeyValues(HashMap<String, OnnxTensor> decoderFeedPast, HashMap<String, OnnxTensor> pastKeyValues, long batch_size) throws OrtException {
        if (pastKeyValues != null && !pastKeyValues.isEmpty()) {
            for (Map.Entry<String, OnnxTensor> entry : pastKeyValues.entrySet()) {
                String key = entry.getKey();
                OnnxTensor value = entry.getValue();
                // 确保只添加有效的past key values
                if (value != null && !value.isClosed()) {
                    decoderFeedPast.put(key, value);
                }
            }
        } else {
            // 初始化past key values
            int num_decoder_layers = (int) config.Config_get("decoder_layers", 12);
            int num_decoder_heads = (int) config.Config_get("decoder_attention_heads", 16);
            int decoder_dim_kv = (int) config.Config_get("d_model", 1024) / num_decoder_heads;
            int num_encoder_heads = (int) config.Config_get("encoder_attention_heads", 16);
            int encoder_dim_kv = (int) config.Config_get("d_model", 1024) / num_encoder_heads;

            long[] encoder_dims = new long[]{batch_size, num_encoder_heads, 0, encoder_dim_kv};
            long[] decoder_dims = new long[]{batch_size, num_decoder_heads, 0, decoder_dim_kv};

            for (int i = 0; i < num_decoder_layers; i++) {
                decoderFeedPast.put(String.format("past_key_values.%d.encoder.key", i),
                        decoder.zeros(encoder_dims, OnnxJavaType.FLOAT));
                decoderFeedPast.put(String.format("past_key_values.%d.encoder.value", i),
                        decoder.zeros(encoder_dims, OnnxJavaType.FLOAT));
                decoderFeedPast.put(String.format("past_key_values.%d.decoder.key", i),
                        decoder.zeros(decoder_dims, OnnxJavaType.FLOAT));
                decoderFeedPast.put(String.format("past_key_values.%d.decoder.value", i),
                        decoder.zeros(decoder_dims, OnnxJavaType.FLOAT));
            }
        }
    }

    private static void closeMapTensors(HashMap<String, OnnxTensor> map) {
        if (map == null || map.isEmpty()) return;
        for (OnnxTensor tensor : map.values()) {
            if (tensor != null && !tensor.isClosed()) {
                try {
                    tensor.close();
                } catch (Exception ignored) {
                }
            }
        }
        map.clear();
    }

    public String translate(String src_text, String src_lang, String tgt_lang, int max_length) throws OrtException {
        int num_beams = (int) config.Config_get("num_beams", 5);
        int top_k = (int) config.Config_get("top_k", 50);
        return translate(src_text, src_lang, tgt_lang, max_length, num_beams, top_k);
    }

    public String translate(String src_text, String src_lang, String tgt_lang, int max_length, int num_beams, int top_k) throws OrtException {
        if (max_length > 200)
            throw new IllegalArgumentException("M2M100-Model: max_length > 200 not support");
        if (Objects.equals(src_text, tgt_lang)) return src_text == null ? "" : src_text;
        Map<String, Object> encoded = tokenizer.callWithSrcLangOnce(src_text, src_lang);
        long[][] input_ids = (long[][]) encoded.get("input_ids");
        long[][] attention_mask = (long[][]) encoded.get("attention_mask");
        OnnxTensor input_ids_Tensor = null;
        OnnxTensor attention_mask_Tensor = null;
        OnnxTensor last_hidden_state_Tensor = null;
        OnnxTensor expanded_encoder_output = null;
        OnnxTensor expanded_attention_mask = null;
        List<Beam> beams = new ArrayList<>();
        List<OnnxTensor> last_decoderKV_1 = new ArrayList<>();
        List<OnnxTensor> last_decoderKV_2 = new ArrayList<>();
        //TMD我真是服了，没人告诉我Encoder的得一直用一开始的啊
        HashMap<String, OnnxTensor> encoder_past_key_values = null;
        long tgt_lang_id = tokenizer.getLangId(tgt_lang);
        ArrayList<Long> initial_input = new ArrayList<>();
        initial_input.add(tokenizer.eosTokenId);
        try {
            input_ids_Tensor = encoder.createLikeArray(input_ids, OnnxJavaType.INT64);
            attention_mask_Tensor = encoder.createLikeArray(attention_mask, OnnxJavaType.INT64);
            Map<String, OnnxTensor> encoder_outputs = encoder.predict(input_ids_Tensor, attention_mask_Tensor);
            last_hidden_state_Tensor = encoder_outputs.get("last_hidden_state");
            beams.add(new Beam(initial_input, 0.0f));
            //输入要扩到num_beams
            long[] encoder_shape = last_hidden_state_Tensor.getInfo().getShape();
            long[] expanded_encoder_shape = new long[]{encoder_shape[0] * num_beams, encoder_shape[1], encoder_shape[2]};
            expanded_encoder_output = encoder.expand(last_hidden_state_Tensor, expanded_encoder_shape);
            long[] attention_mask_shape = attention_mask_Tensor.getInfo().getShape();
            long[] expanded_attention_mask_shape = new long[]{attention_mask_shape[0] * num_beams, attention_mask_shape[1]};
            expanded_attention_mask = encoder.expand(attention_mask_Tensor, expanded_attention_mask_shape);
            for (int step = 0; step < max_length; step++) {
                List<Beam> next_beams = new ArrayList<>();
                for (Beam beam : beams) {
                    if (beam.isFinished()) {
                        next_beams.add(beam);
                        continue;
                    }
                    OnnxTensor decoder_input = null;
                    try {
                        decoder_input = decoder.tokens2Tensor(beam.getSequence());
                        long[] decoder_input_shape = decoder_input.getInfo().getShape();
                        long[] expanded_decoder_input_shape = new long[]{decoder_input_shape[0] * num_beams, decoder_input_shape[1]};
                        OnnxTensor expanded_decoder_input = encoder.expand(decoder_input, expanded_decoder_input_shape);
                        decoder_input.close();
                        decoder_input = expanded_decoder_input;
                        HashMap<String, OnnxTensor> past_key_values = new HashMap<>();
                        if (step == 0) {
                            // 第一次调用，初始化encoder的key和value
                            addPastKeyValues(past_key_values, null, decoder_input.getInfo().getShape()[0]);
                        } else {
                            // 后续步骤，使用之前保存的encoder key和value
                            if (encoder_past_key_values != null) {
                                for (Map.Entry<String, OnnxTensor> entry : encoder_past_key_values.entrySet()) {
                                    if (entry.getKey().contains(".encoder.")) {
                                        past_key_values.put(entry.getKey(), entry.getValue());
                                    }
                                }
                            }
                            // 添加decoder的past key values
                            addPastKeyValues(past_key_values, beam.getPastKeyValues(), decoder_input.getInfo().getShape()[0]);
                        }
                        HashMap<String, OnnxTensor> decoder_outputs = decoder.predict(
                                decoder_input, expanded_attention_mask, expanded_encoder_output,
                                step > 0, past_key_values
                        );
                        OnnxTensor logits_tensor = decoder_outputs.get("logits");
                        float[][][] logits = (float[][][]) logits_tensor.getValue();
                        float[] next_token_logits = logits[0][logits[0].length - 1];
                        if (step == 0) {
                            encoder_past_key_values = new HashMap<>();
                            for (Map.Entry<String, OnnxTensor> entry : decoder_outputs.entrySet()) {
                                String key = entry.getKey();
                                if (key.startsWith("present.") && key.contains(".encoder.")) {
                                    String past_key = key.replace("present.", "past_key_values.");
                                    encoder_past_key_values.put(past_key, entry.getValue());
                                }
                            }
                            // 强制选择目标语言token
                            for (int i = 0; i < next_token_logits.length; i++) {
                                if (i == tgt_lang_id) {
                                    next_token_logits[i] = 0.0f;
                                } else {
                                    next_token_logits[i] = Float.NEGATIVE_INFINITY;
                                }
                            }
                        } else {
                            //后续的Encoder不需要
                            for (Map.Entry<String, OnnxTensor> entry : decoder_outputs.entrySet()) {
                                String key = entry.getKey();
                                if (key.startsWith("present.") && key.contains(".encoder.")) {
                                    entry.getValue().close();
                                }
                            }
                        }

                        float[] probs = TensorBase.softmax(next_token_logits);
                        int[] topk_indices = TensorBase.topK(probs, top_k);
                        HashMap<String, OnnxTensor> new_past = new HashMap<>();
                        for (Map.Entry<String, OnnxTensor> entry : decoder_outputs.entrySet()) {
                            String key = entry.getKey();
                            if (key.startsWith("present.")) {
                                String past_key = key.replace("present.", "past_key_values.");
                                // 跳过encoder部分，因为我们已经单独保存了
                                if (!past_key.contains(".encoder.")) {
                                    new_past.put(past_key, entry.getValue());
                                    if (step % 2 == 0)//两个decoder交替负责清理，保证没有内存泄漏
                                        last_decoderKV_1.add(entry.getValue());
                                    else
                                        last_decoderKV_2.add(entry.getValue());
                                }
                            } else if (!key.equals("logits")) {
                                entry.getValue().close();
                            }
                        }
                        logits_tensor.close();
                        for (int idx : topk_indices) {
                            ArrayList<Long> new_sequence = new ArrayList<>(beam.getSequence());
                            new_sequence.add((long) idx);
                            float new_score = beam.getScore() + (float) Math.log(probs[idx]);
                            HashMap<String, OnnxTensor> beam_past = new HashMap<>(new_past);
                            next_beams.add(new Beam(new_sequence, new_score, beam_past));
                        }
                    } finally {
                        if (decoder_input != null && !decoder_input.isClosed()) {
                            decoder_input.close();
                        }
                    }
                }
                for (Beam beam : beams) {
                    if (!next_beams.contains(beam)) {
                        beam.close();
                    }
                }
                if (step % 2 == 0) {
                    for (OnnxTensor tensor : last_decoderKV_2) {
                        if (!tensor.isClosed())
                            tensor.close();
                    }
                    last_decoderKV_2.clear();
                } else {
                    for (OnnxTensor tensor : last_decoderKV_1) {
                        if (!tensor.isClosed())
                            tensor.close();
                    }
                    last_decoderKV_1.clear();
                }
                beams = selectTopBeams(next_beams, num_beams);
                boolean all_finished = true;
                for (Beam beam : beams) {
                    if (!beam.isFinished()) {
                        all_finished = false;
                        break;
                    }
                }
                if (all_finished) break;
            }
            Beam best_beam = beams.get(0);
            for (Beam beam : beams) {
                if (beam.getScore() > best_beam.getScore()) {
                    best_beam = beam;
                }
            }
            return tokenizer.decode(best_beam.getSequence(), false);
        } finally {
            if (input_ids_Tensor != null) input_ids_Tensor.close();
            if (attention_mask_Tensor != null) attention_mask_Tensor.close();
            if (last_hidden_state_Tensor != null) last_hidden_state_Tensor.close();
            if (expanded_encoder_output != null) expanded_encoder_output.close();
            if (expanded_attention_mask != null) expanded_attention_mask.close();
            for (Beam beam : beams) {
                beam.close();
            }
            if (encoder_past_key_values != null) {
                closeMapTensors(encoder_past_key_values);
            }
        }
    }

    class Beam {
        private final ArrayList<Long> sequence;
        private final float score;
        private HashMap<String, OnnxTensor> pastKeyValues;
        private final boolean finished;

        public void close() {
            if (pastKeyValues != null) {
                closeMapTensors(pastKeyValues);
                pastKeyValues = null;
            }
        }

        public Beam(ArrayList<Long> sequence, float score) {
            this.sequence = sequence;
            this.score = score;
            this.pastKeyValues = null;
            this.finished = false;
        }

        public Beam(ArrayList<Long> sequence, float score, HashMap<String, OnnxTensor> pastKeyValues) {
            this.sequence = sequence;
            this.score = score;
            this.pastKeyValues = pastKeyValues;
            this.finished = sequence.get(sequence.size() - 1) == tokenizer.eosTokenId;
        }

        public boolean isFinished() {
            return finished;
        }

        public ArrayList<Long> getSequence() {
            return sequence;
        }

        public float getScore() {
            return score;
        }

        public HashMap<String, OnnxTensor> getPastKeyValues() {
            return pastKeyValues;
        }


    }

    private static List<Beam> selectTopBeams(List<Beam> beams, int num_beams) {
        Collections.sort(beams, new Comparator<Beam>() {
            @Override
            public int compare(Beam a, Beam b) {
                return Float.compare(b.getScore(), a.getScore());
            }
        });
        return beams.subList(0, Math.min(num_beams, beams.size()));
    }
}