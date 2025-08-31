package com.nwdxlgzs.translate;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Set;

import org.json.JSONArray;
import org.json.JSONObject;
import org.json.JSONException;

public class M2M100TokenizerFast {
    private JSONObject cfg;
    private Map<String, Long> vocab;
    private Map<Long, String> idsToTokens;
    public long unkTokenId;
    public long padTokenId;
    public long bosTokenId;
    public long eosTokenId;
    private Map<String, Long> langCodeToId;
    private List<String[]> merges;
    private Map<String, Long> bpeRanks;
    private Set<String> specialStrs;
    private String srcLang;

    public M2M100TokenizerFast(String tokenizerJson) throws IOException, JSONException {
        // 读取 JSON 文件
        String tokenizerJsonContent;
        try (FileInputStream fis = new FileInputStream(tokenizerJson)) {
            int len = fis.available();
            byte[] bytes = new byte[len];
            byte[] buf = new byte[128 * 1024];
            int total = 0;
            int r;
            while ((r = fis.read(buf)) != -1) {
                System.arraycopy(buf, 0, bytes, total, r);
                total += r;
            }
            tokenizerJsonContent = new String(bytes);
        }
        this.cfg = new JSONObject(tokenizerJsonContent);

        // 初始化词汇表
        JSONObject model = cfg.getJSONObject("model");
        JSONObject vocabJson = model.getJSONObject("vocab");
        this.vocab = new HashMap<>();
        this.idsToTokens = new HashMap<>();

        // 使用迭代器而不是增强for循环
        Iterator<String> keys = vocabJson.keys();
        while (keys.hasNext()) {
            String key = keys.next();
            long value = vocabJson.getLong(key);
            vocab.put(key, value);
            idsToTokens.put(value, key);
        }

        // 初始化特殊标记
        this.langCodeToId = new HashMap<>();
        this.specialStrs = new HashSet<>();
        JSONArray addedTokens = cfg.optJSONArray("added_tokens");
        if (addedTokens != null) {
            for (int i = 0; i < addedTokens.length(); i++) {
                JSONObject token = addedTokens.getJSONObject(i);
                String contentStr = token.getString("content");
                long id = token.getLong("id");

                if ("<unk>".equals(contentStr)) this.unkTokenId = id;
                if ("<pad>".equals(contentStr)) this.padTokenId = id;
                if ("<s>".equals(contentStr)) this.bosTokenId = id;
                if ("</s>".equals(contentStr)) this.eosTokenId = id;

                if (contentStr.startsWith("__") && contentStr.endsWith("__")) {
                    langCodeToId.put(contentStr, id);
                }
                if (token.optBoolean("special", false)) {
                    specialStrs.add(contentStr);
                }
            }
        }

        // 初始化合并规则
        JSONArray mergesArray = model.getJSONArray("merges");
        this.merges = new ArrayList<>();
        this.bpeRanks = new HashMap<>();
        for (int i = 0; i < mergesArray.length(); i++) {
            String merge = mergesArray.getString(i);
            String[] parts = merge.split(" ");
            merges.add(parts);
            bpeRanks.put(parts[0] + " " + parts[1], (long) i);
        }
    }

    public String getSrcLang() {
        return srcLang;
    }

    public void setSrcLang(String lang) {
        this.srcLang = lang;
    }

    public long getLangId(String lang) {
        String code = "__" + lang + "__";
        if (!langCodeToId.containsKey(code)) {
            throw new NoSuchElementException("Language not found: " + lang);
        }
        return langCodeToId.get(code);
    }

    public Map<String, Object> callWithSrcLangOnce(String text, String srcLang) {
        List<String> texts = new ArrayList<>();
        texts.add(text);
        return call(texts, srcLang);
    }

    public Map<String, Object> call(String text) {
        List<String> texts = new ArrayList<>();
        texts.add(text);
        return call(texts, srcLang);
    }

    public Map<String, Object> call(List<String> texts, String srcLang) {
        if (texts.size() != 1)
            throw new UnsupportedOperationException("Batch processing not implemented");
        String text = texts.get(0);
        if (srcLang == null)
            throw new IllegalStateException("srcLang not set");
        long langId = getLangId(srcLang);
        List<Long> tokenIds = tokenize(text);
        long[] inputIds = new long[tokenIds.size() + 2];
        inputIds[0] = langId;
        for (int i = 0; i < tokenIds.size(); i++) {
            inputIds[i + 1] = tokenIds.get(i);
        }
        inputIds[inputIds.length - 1] = eosTokenId;
        long[] attentionMask = new long[inputIds.length];
        Arrays.fill(attentionMask, 1);
        Map<String, Object> result = new HashMap<>();
        result.put("input_ids", new long[][]{inputIds});
        result.put("attention_mask", new long[][]{attentionMask});
        return result;
    }

    public String decode(long[] sequence, boolean skipSpecialTokens) {
        List<String> tokens = new ArrayList<>();
        for (long id : sequence) {
            tokens.add(idsToTokens.get(id));
        }
        if (skipSpecialTokens) {
            List<String> toRemove = new ArrayList<>();
            for (String token : tokens) {
                if (specialStrs.contains(token)) {
                    toRemove.add(token);
                }
            }
            tokens.removeAll(toRemove);
        }
        StringBuilder joined = new StringBuilder();
        for (String token : tokens) {
            joined.append(token);
        }
        return joined.toString().replace("▁", " ").trim();
    }

    public String decode(List<Long> sequence, boolean skipSpecialTokens) {
        List<String> tokens = new ArrayList<>();
        for (long id : sequence) {
            tokens.add(idsToTokens.get(id));
        }
        if (skipSpecialTokens) {
            List<String> toRemove = new ArrayList<>();
            for (String token : tokens) {
                if (specialStrs.contains(token)) {
                    toRemove.add(token);
                }
            }
            tokens.removeAll(toRemove);
        }
        StringBuilder joined = new StringBuilder();
        for (String token : tokens) {
            joined.append(token);
        }
        return joined.toString().replace("▁", " ").trim();
    }

    public List<String> batchDecode(long[][] sequences, boolean skipSpecialTokens) {
        List<String> results = new ArrayList<>();
        for (long[] sequence : sequences) {
            results.add(decode(sequence, skipSpecialTokens));
        }
        return results;
    }

    public List<String> batchDecode(List<List<Long>> sequences, boolean skipSpecialTokens) {
        List<String> results = new ArrayList<>();
        for (List<Long> sequence : sequences) {
            results.add(decode(sequence, skipSpecialTokens));
        }
        return results;
    }

    private List<Long> tokenize(String text) {
        String[] words = text.trim().split(" ");
        List<Long> ids = new ArrayList<>();
        for (String w : words) {
            if (w.isEmpty()) continue;
            ids.addAll(bpeWord("▁" + w));
        }
        return ids;
    }

    private List<Long> bpeWord(String word) {
        List<String> characters = new ArrayList<>();
        for (int i = 0; i < word.length(); i++) {
            characters.add(String.valueOf(word.charAt(i)));
        }

        Set<String[]> pairs = getPairs(characters);
        if (pairs.isEmpty()) {
            List<Long> result = new ArrayList<>();
            Long id = vocab.get(characters.get(0));
            if (id == null) {
                id = unkTokenId;
            }
            result.add(id);
            return result;
        }

        while (true) {
            // 找到优先级最高的对
            String[] bigram = null;
            long minRank = Long.MAX_VALUE;
            for (String[] pair : pairs) {
                String key = pair[0] + " " + pair[1];
                Long rank = bpeRanks.get(key);
                if (rank == null) {
                    rank = Long.MAX_VALUE;
                }
                if (rank < minRank) {
                    minRank = rank;
                    bigram = pair;
                }
            }

            if (bigram == null || minRank == Long.MAX_VALUE) {
                break;
            }

            String first = bigram[0];
            String second = bigram[1];
            List<String> newWord = new ArrayList<>();
            int i = 0;

            while (i < characters.size()) {
                int j = indexOf(characters, first, i);
                if (j == -1) {
                    for (int k = i; k < characters.size(); k++) {
                        newWord.add(characters.get(k));
                    }
                    break;
                }
                for (int k = i; k < j; k++) {
                    newWord.add(characters.get(k));
                }
                i = j;

                if (i < characters.size() - 1 &&
                        characters.get(i).equals(first) &&
                        characters.get(i + 1).equals(second)) {
                    newWord.add(first + second);
                    i += 2;
                } else {
                    newWord.add(characters.get(i));
                    i += 1;
                }
            }

            characters = newWord;
            if (characters.size() == 1) break;
            pairs = getPairs(characters);
        }

        List<Long> wordIds = new ArrayList<>();
        for (String token : characters) {
            Long id = vocab.get(token);
            if (id == null) {
                id = unkTokenId;
            }
            wordIds.add(id);
        }
        return wordIds;
    }

    private int indexOf(List<String> list, String value, int fromIndex) {
        for (int i = fromIndex; i < list.size(); i++) {
            if (list.get(i).equals(value)) {
                return i;
            }
        }
        return -1;
    }

    private Set<String[]> getPairs(List<String> word) {
        Set<String[]> pairs = new HashSet<>();
        for (int i = 0; i < word.size() - 1; i++) {
            pairs.add(new String[]{word.get(i), word.get(i + 1)});
        }
        return pairs;
    }

    public static M2M100TokenizerFast fromPretrained(String path) throws IOException, JSONException {
        return new M2M100TokenizerFast(new File(path, "tokenizer.json").getPath());
    }
}