package com.nwdxlgzs.translate;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.FileInputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

public class M2MConfig extends HashMap<String, Object> {

    public M2MConfig(String config_json, String generation_config_json) {
        loadJson(config_json);
        loadJson(generation_config_json);
        autoConfig();
    }

    private void autoConfig() {
        Config_putIfAbsent("max_length", 20);
        Config_putIfAbsent("max_new_tokens", null);
        Config_putIfAbsent("min_length", 0);
        Config_putIfAbsent("min_new_tokens", null);
        Config_putIfAbsent("early_stopping", false);
        Config_putIfAbsent("max_time", null);
        Config_putIfAbsent("do_sample", false);
        Config_putIfAbsent("num_beams", 1);
        Config_putIfAbsent("num_beam_groups", 1);
        Config_putIfAbsent("penalty_alpha", null);
        Config_putIfAbsent("use_cache", true);
        Config_putIfAbsent("temperature", 1.0);
        Config_putIfAbsent("top_k", 50);
        Config_putIfAbsent("top_p", 1.0);
        Config_putIfAbsent("typical_p", 1.0);
        Config_putIfAbsent("epsilon_cutoff", 0.0);
        Config_putIfAbsent("eta_cutoff", 0.0);
        Config_putIfAbsent("diversity_penalty", 0.0);
        Config_putIfAbsent("repetition_penalty", 1.0);
        Config_putIfAbsent("encoder_repetition_penalty", 1.0);
        Config_putIfAbsent("length_penalty", 1.0);
        Config_putIfAbsent("no_repeat_ngram_size", 0);
        Config_putIfAbsent("bad_words_ids", null);
        Config_putIfAbsent("force_words_ids", null);
        Config_putIfAbsent("renormalize_logits", false);
        Config_putIfAbsent("constraints", null);
        Config_putIfAbsent("forced_bos_token_id", null);
        Config_putIfAbsent("forced_eos_token_id", null);
        Config_putIfAbsent("remove_invalid_values", false);
        Config_putIfAbsent("exponential_decay_length_penalty", null);
        Config_putIfAbsent("suppress_tokens", null);
        Config_putIfAbsent("begin_suppress_tokens", null);
        Config_putIfAbsent("forced_decoder_ids", null);
        Config_putIfAbsent("num_return_sequences", 1);
        Config_putIfAbsent("output_attentions", false);
        Config_putIfAbsent("output_hidden_states", false);
        Config_putIfAbsent("output_scores", false);
        Config_putIfAbsent("return_dict_in_generate", false);
        Config_putIfAbsent("pad_token_id", null);
        Config_putIfAbsent("bos_token_id", null);
        Config_putIfAbsent("eos_token_id", null);
        Config_putIfAbsent("encoder_no_repeat_ngram_size", 0);
        Config_putIfAbsent("decoder_start_token_id", null);
        Config_putIfAbsent("generation_kwargs", new ArrayList<>());
    }

    public void Config_putIfAbsent(String key, Object defaultValue) {
        if (!containsKey(key)) {
            put(key, defaultValue);
        }
    }

    public Object Config_get(String key, Object defaultValue) {
        if (!containsKey(key)) {
            return get(key);
        }
        return defaultValue;
    }

    private static Object toJavaObject(Object json) throws JSONException {
        if (json == null) return null;
        if (json instanceof JSONObject) {
            Map<String, Object> map = new HashMap<>();
            JSONObject jo = (JSONObject) json;
            Iterator<String> keys = jo.keys();
            while (keys.hasNext()) {
                String k = keys.next();
                map.put(k, toJavaObject(jo.get(k)));
            }
            return map;
        }
        if (json instanceof JSONArray) {
            List<Object> list = new java.util.ArrayList<>();
            JSONArray ja = (JSONArray) json;
            for (int i = 0; i < ja.length(); i++) {
                list.add(toJavaObject(ja.get(i)));
            }
            return list;
        }
        return json;
    }

    private void loadJson(String json_path) {
        try {
            FileInputStream fis = new FileInputStream(json_path);
            int len = fis.available();
            byte[] bytes = new byte[len];
            byte[] buf = new byte[128 * 1024];
            int total = 0;
            int r;
            while ((r = fis.read(buf)) != -1) {
                System.arraycopy(buf, 0, bytes, total, r);
                total += r;
            }
            String content = new String(bytes);
            JSONObject src = new JSONObject(content);
            Iterator<String> keys = src.keys();
            while (keys.hasNext()) {
                String k = keys.next();
                put(k, toJavaObject(src.get(k)));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
