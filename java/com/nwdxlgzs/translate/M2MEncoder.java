package com.nwdxlgzs.translate;

import android.content.res.AssetManager;

import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Collections;
import java.util.HashMap;
import java.util.Set;

import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;

public class M2MEncoder extends TensorBase {
    private volatile OrtSession session;
    private final String modelPath;

    public M2MEncoder(OrtEnvironment env, String encoder_onnx) {
        super(env);
        modelPath = encoder_onnx;
    }

    public HashMap<String, OnnxTensor> predict(OnnxTensor input_ids, OnnxTensor attention_mask) throws OrtException {
//        long st=System.currentTimeMillis();
        OrtSession ortSession = getSession();
        HashMap<String, OnnxTensor> inputs = new HashMap<>();
        for (String name : ortSession.getInputNames()) {
            if ("input_ids".equals(name)) {
                inputs.put(name, input_ids);
            } else if ("attention_mask".equals(name)) {
                inputs.put(name, attention_mask);
            }
        }
        OrtSession.Result result = ortSession.run(inputs);
        HashMap<String, OnnxTensor> outputs = new HashMap<>();
        OnnxValue value = result.get(0);
        outputs.put("last_hidden_state",(OnnxTensor)value);
        // result.close();last_hidden_state会被使用，所以不close
//        System.out.println("ENCODER = " + (System.currentTimeMillis()-st) + "ms");
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
                        session =  ortEnv.createSession(modelPath, options);
                    } catch (Exception e) {
                        throw new RuntimeException("Failed to create session", e);
                    }
                }
            }
        }
        return session;
    }


}
