Do something like what Optimum does, but for ONNX on Android

# Models(encoder_model.onnx+decoder_model_merged.onnx)

## M2M100
[https://huggingface.co/Xenova/m2m100_418M](https://huggingface.co/Xenova/m2m100_418M)
## NLLB-200
[https://huggingface.co/Xenova/nllb-200-distilled-600M](https://huggingface.co/Xenova/nllb-200-distilled-600M)
## MBART
[https://huggingface.co/Xenova/mbart-large-50-many-to-many-mmt](https://huggingface.co/Xenova/mbart-large-50-many-to-many-mmt)

# Activity
> Part.A

```sh
src\main\assets>ls
config.json  decoder_model_merged.onnx  encoder_model.onnx  generation_config.json  tokenizer.json
```

> Part.B
```java
package com.nwdxlgzs;

import android.app.Activity;
import android.app.ProgressDialog;
import android.content.Context;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.widget.Toast;

import com.nwdxlgzs.translate.Polyglots;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

import ai.onnxruntime.OrtEnvironment;

public class MainActivity extends Activity {

    private static final String TAG = "MainActivity";
    private static final String PREFS_KEY = "LastUnzipTime";

    private final String[] fileTargets = {
            "config.json",
            "decoder_model_merged.onnx",
            "encoder_model.onnx",
            "generation_config.json",
            "tokenizer.json"
    };

    private OrtEnvironment ortEnv;
    private File modelDir;
    private final ExecutorService executor = Executors.newFixedThreadPool(3);
    private final Handler mainHandler = new Handler(Looper.getMainLooper());

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        ortEnv = OrtEnvironment.getEnvironment();
        modelDir = getFilesDir();

        if (shouldUnzipModels()) {
            unzipModelsAsync();
        } else {
            initPolyglots();
        }
    }

    private boolean shouldUnzipModels() {
        try {
            PackageInfo pi = getPackageManager()
                    .getPackageInfo(getPackageName(), 0);
            long lastInstallTime = pi.lastUpdateTime;
            long lastUnzipTime = getPreferences(Context.MODE_PRIVATE)
                    .getLong(PREFS_KEY, 0);
            return lastInstallTime > lastUnzipTime;
        } catch (PackageManager.NameNotFoundException e) {
            Log.e(TAG, "shouldUnzipModels", e);
            return true;
        }
    }

    private void unzipModelsAsync() {
        ProgressDialog dialog = new ProgressDialog(this);
        dialog.setMessage("Unzipping models...");
        dialog.setProgressStyle(ProgressDialog.STYLE_HORIZONTAL);
        dialog.setMax(fileTargets.length);
        dialog.setCancelable(false);
        dialog.show();

        AtomicInteger completed = new AtomicInteger(0);
        for (String name : fileTargets) {
            executor.execute(() -> {
                File outFile = new File(modelDir, name);
                copyAssetToFile(name, outFile);
                int progress = completed.incrementAndGet();
                mainHandler.post(() -> {
                    dialog.setProgress(progress);
                    if (progress == fileTargets.length) {
                        dialog.dismiss();
                        markUnzipDone();
                        initPolyglots();
                    }
                });
            });
        }
    }

    private void copyAssetToFile(String assetName, File outFile) {
        outFile.delete();
        AssetManager am = getAssets();
        try (InputStream in = am.open(assetName);
             OutputStream out = new FileOutputStream(outFile)) {
            byte[] buf = new byte[8192];
            int len;
            while ((len = in.read(buf)) != -1) {
                out.write(buf, 0, len);
            }
        } catch (IOException e) {
            Log.e(TAG, "copyAssetToFile: " + assetName, e);
        }
    }

    private void markUnzipDone() {
        getPreferences(Context.MODE_PRIVATE)
                .edit()
                .putLong(PREFS_KEY, System.currentTimeMillis())
                .apply();
    }

    private void initPolyglots() {
        try {
            Polyglots polyglots = new Polyglots(ortEnv, modelDir.getAbsolutePath());
            String result = polyglots.translate("生活就像一盒巧克力。", "zh", "en", 200);
            //预热完毕，用最低的束（贪婪）搜索
            long st=System.currentTimeMillis();
            result = polyglots.translate("生活就像一盒巧克力。", "zh", "en", 200,1,1);
            System.out.println("polyglots.translate = " + (System.currentTimeMillis()-st) + "ms");
            Log.i(TAG, "translate result = " + result);
        } catch (Exception e) {
            Log.e(TAG, "initPolyglots", e);
            mainHandler.post(() ->
                    Toast.makeText(this, "模型初始化失败: " + e.getMessage(), Toast.LENGTH_LONG).show());
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        executor.shutdownNow();
    }
}
```
