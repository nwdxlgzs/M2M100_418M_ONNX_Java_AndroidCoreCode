package com.nwdxlgzs.translate;

import static ai.onnxruntime.OnnxJavaType.BOOL;

import android.content.res.AssetManager;

import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;

public class M2MDecoder extends TensorBase {
    private volatile OrtSession session;
    private final String modelPath;

    public M2MDecoder(OrtEnvironment env, String encoder_onnx) {
        super(env);
        modelPath = encoder_onnx;
    }

    public HashMap<String, OnnxTensor> predict(OnnxTensor input_ids,
                                               OnnxTensor encoder_attention_mask,
                                               OnnxTensor encoder_hidden_states,
                                               boolean use_cache_branch,
                                               HashMap<String, OnnxTensor> past_key_values) throws OrtException {
//        long st=System.currentTimeMillis();
        OrtSession ortSession = getSession();
        HashMap<String, OnnxTensor> inputs = new HashMap<>();
        OnnxTensor use_cache_branch_Tensor = createLikeArray(new boolean[]{use_cache_branch});
        for (String name : ortSession.getInputNames()) {
            if ("input_ids".equals(name)) {
                inputs.put(name, input_ids);
            } else if ("encoder_attention_mask".equals(name)) {
                inputs.put(name, encoder_attention_mask);
            } else if ("use_cache_branch".equals(name)) {
                inputs.put(name, use_cache_branch_Tensor);
            } else if ("encoder_hidden_states".equals(name)) {
                inputs.put(name, encoder_hidden_states);
            } else {
                if (past_key_values.containsKey(name)) {
                    inputs.put(name, past_key_values.get(name));
                }
            }
        }
        OrtSession.Result result = ortSession.run(inputs);
        HashMap<String, OnnxTensor> outputs = new HashMap<>();
        for (Map.Entry<String, OnnxValue> entry : result) {
            String name = entry.getKey();
            OnnxValue value = entry.getValue();
            if (value instanceof OnnxTensor) {
                outputs.put(name, (OnnxTensor) value);
            } else {
                value.close();
//                System.err.println("Skipping non-tensor output: " + name + " (" + value.getType() + ")");
            }
        }
        use_cache_branch_Tensor.close();
//        System.out.println("DECODER = " + (System.currentTimeMillis()-st) + "ms");
        return outputs;
    }

    private OrtSession getSession() {
        if (session == null) {
            synchronized (this) {
                if (session == null) {
                    try {
                        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
                        options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
                        try {
                            options.addNnapi();
                        } catch (OrtException e) {
                            options.addCPU(true);
                        }
                        session = ortEnv.createSession(modelPath, options);
                    } catch (Exception e) {
                        throw new RuntimeException("Failed to create session", e);
                    }
                }
            }
        }
        return session;
    }


}
