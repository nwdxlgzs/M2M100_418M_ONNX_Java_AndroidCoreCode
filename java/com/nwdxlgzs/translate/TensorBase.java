package com.nwdxlgzs.translate;

import static ai.onnxruntime.OnnxJavaType.*;

import ai.onnxruntime.*;

import java.lang.reflect.Array;
import java.nio.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class TensorBase {
    public static long[] getArrayShape(Object array) {
        if (array == null || !array.getClass().isArray()) {
            return new long[0];
        }
        List<Long> shapeList = new ArrayList<>();
        Object current = array;
        while (current.getClass().isArray()) {
            int length = Array.getLength(current);
            shapeList.add((long) length);
            if (length > 0) {
                Object firstElement = Array.get(current, 0);
                if (firstElement != null && firstElement.getClass().isArray()) {
                    current = firstElement;
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        long[] shape = new long[shapeList.size()];
        for (int i = 0; i < shapeList.size(); i++) {
            shape[i] = shapeList.get(i);
        }
        return shape;
    }

    public final OrtEnvironment ortEnv;

    public TensorBase(OrtEnvironment ortEnv) {
        this.ortEnv = ortEnv;
    }

    public static float[] softmax(float[] logits) {
        float max = Float.NEGATIVE_INFINITY;
        for (float value : logits) {
            if (value > max) max = value;
        }

        float sum = 0.0f;
        float[] exp = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            exp[i] = (float) Math.exp(logits[i] - max);
            sum += exp[i];
        }

        for (int i = 0; i < exp.length; i++) {
            exp[i] /= sum;
        }
        return exp;
    }

    public static int[] topK(float[] values, int k) {
        int n = values.length;
        k = Math.min(k, n);
        int[] idx = new int[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        for (int i = 0; i < k; i++) {
            int max = i;
            for (int j = i + 1; j < n; j++) {
                if (values[idx[j]] > values[idx[max]]) {
                    max = j;
                }
            }
            int tmp = idx[i];
            idx[i] = idx[max];
            idx[max] = tmp;
        }
        int[] result = new int[k];
        System.arraycopy(idx, 0, result, 0, k);
        return result;
    }


    public OnnxTensor expand(OnnxTensor src, long[] newShape) throws OrtException {
        long[] oldShape = src.getInfo().getShape();
        if (oldShape.length != newShape.length) {
            throw new OrtException("Rank mismatch: " + oldShape.length + " vs " + newShape.length);
        }
        long repeat = newShape[0] / oldShape[0];
        if (newShape[0] % oldShape[0] != 0) {
            throw new OrtException("Dimension 0 not divisible: " + newShape[0] + "/" + oldShape[0]);
        }
        for (int i = 1; i < oldShape.length; i++) {
            if (oldShape[i] != newShape[i]) {
                throw new OrtException("Dimension " + i + " changed: " + oldShape[i] + " -> " + newShape[i]);
            }
        }
        Object oldData = src.getValue();
        int[] intShape = new int[newShape.length];
        for (int i = 0; i < newShape.length; i++) {
            intShape[i] = Math.toIntExact(newShape[i]);
        }
        Object newData = Array.newInstance(
                leafComponentType(oldData.getClass()),
                intShape);
        expandRecursive(oldData, newData, repeat);
        return OnnxTensor.createTensor(ortEnv, newData);
    }

    private static Class<?> leafComponentType(Class<?> arrayClass) {
        Class<?> c = arrayClass;
        while (c.isArray()) {
            c = c.getComponentType();
        }
        return c;
    }

    /** 递归拷贝：把 src 沿第一维复制 repeat 次到 dst */
    private static void expandRecursive(Object src, Object dst, long repeat) {
        int len = Array.getLength(src);          // 当前这一维的长度
        if (src.getClass().getComponentType().isArray()) {
            // 还没到最底层，继续递归
            for (int r = 0; r < repeat; r++) {
                for (int i = 0; i < len; i++) {
                    Object srcSub = Array.get(src, i);
                    Object dstSub = Array.get(dst, r * len + i);
                    expandRecursive(srcSub, dstSub, 1); // 下一层不再重复，只拷贝
                }
            }
        } else {
            // 最底层 1-D 基本类型数组，整段拷贝
            for (int r = 0; r < repeat; r++) {
                System.arraycopy(src, 0, dst, r * len, len);
            }
        }
    }


    public OnnxTensor tokens2Tensor(List<Long> tokens) throws OrtException {
        long[] arr = new long[tokens.size()];
        for (int i = 0; i < tokens.size(); i++) {
            arr[i] = tokens.get(i);
        }
        return createLikeArray(new long[][]{arr}, INT64);
    }

    public OnnxTensor createLikeArray(Object data) throws OrtException {
        Class<?> arrayClass = data.getClass();
        while (arrayClass.isArray()) {
            arrayClass = arrayClass.getComponentType();
        }
        return createLikeArray(data, autoType(arrayClass));
    }

    public OnnxTensor createLikeArray(Object data, OnnxJavaType type) throws OrtException {
        return create(data, getArrayShape(data), type);
    }

    public OnnxTensor create(ByteBuffer data, long[] shape, OnnxJavaType type) throws OrtException {
        return OnnxTensor.createTensor(ortEnv, data, shape, type);
    }


    public OnnxTensor create(Object array, long[] shape, OnnxJavaType type) throws OrtException {
        if (!isArrayTypeCompatible(array, type)) {
            throw new IllegalArgumentException("Array type does not match specified OnnxJavaType");
        }
        ByteBuffer buf = toByteBuffer(array, type);
        return create(buf, shape, type);
    }

    public static OnnxJavaType autoType(Class<?> cls) {
        if (cls == float.class) return FLOAT;
        else if (cls == double.class) return DOUBLE;
        else if (cls == byte.class) return INT8;
        else if (cls == short.class) return INT16;
        else if (cls == int.class) return INT32;
        else if (cls == long.class) return INT64;
        else if (cls == boolean.class) return BOOL;
        return UNKNOWN;
    }

    private boolean isArrayTypeCompatible(Object array, OnnxJavaType type) {
        Class<?> arrayClass = leafComponentType(array.getClass());
        switch (type) {
            case FLOAT:
                return arrayClass == float.class;
            case DOUBLE:
                return arrayClass == double.class;
            case INT8:
            case UINT8:
                return arrayClass == byte.class;
            case INT16:
            case FLOAT16:
            case BFLOAT16:
                return arrayClass == short.class;
            case INT32:
                return arrayClass == int.class;
            case INT64:
                return arrayClass == long.class;
            case BOOL:
                return arrayClass == boolean.class;
            default:
                return false;
        }
    }


    public OnnxTensor zeros(long[] shape, OnnxJavaType type) throws OrtException {
        int bytes = (int) (numElements(shape) * type.size);
        ByteBuffer buf = ByteBuffer.allocateDirect(bytes)
                .order(ByteOrder.nativeOrder());
        return create(buf, shape, type);
    }

    public OnnxTensor ones(long[] shape, OnnxJavaType type) throws OrtException {
        int n = (int) numElements(shape);
        ByteBuffer buf = ByteBuffer.allocateDirect(n * type.size)
                .order(ByteOrder.nativeOrder());
        switch (type) {
            case FLOAT:
                for (int i = 0; i < n; i++) buf.putFloat(1f);
                break;
            case DOUBLE:
                for (int i = 0; i < n; i++) buf.putDouble(1d);
                break;
            case INT8:
            case UINT8:
            case BOOL:
                for (int i = 0; i < n; i++) buf.put((byte) 1);
                break;
            case INT16:
                for (int i = 0; i < n; i++) buf.putShort((short) 1);
                break;
            case INT32:
                for (int i = 0; i < n; i++) buf.putInt(1);
                break;
            case INT64:
                for (int i = 0; i < n; i++) buf.putLong(1L);
                break;
            case FLOAT16:
                for (int i = 0; i < n; i++) buf.putShort((short) 0x3C00);
                break;
            case BFLOAT16:
                for (int i = 0; i < n; i++) buf.putShort((short) 0x3F80);
            default:
                throw new IllegalArgumentException("Unsupported type " + type);
        }
        buf.rewind();
        return create(buf, shape, type);
    }

    public OnnxTensor zerosLike(OnnxTensor src) throws OrtException {
        return zeros(src.getInfo().getShape(), src.getInfo().type);
    }

    public OnnxTensor onesLike(OnnxTensor src) throws OrtException {
        return ones(src.getInfo().getShape(), src.getInfo().type);
    }


    public OnnxTensor flat(OnnxTensor src) throws OrtException {
        long[] shape = src.getInfo().getShape();
        long total = numElements(shape);
        return create(toByteBuffer(src), new long[]{total}, src.getInfo().type);
    }


    public OnnxTensor slice(OnnxTensor src, long[] starts, long[] ends) throws OrtException {
        long[] srcShape = src.getInfo().getShape();
        int dims = srcShape.length;
        if (starts.length != dims || ends.length != dims)
            throw new IllegalArgumentException("starts/ends mismatch shape");
        for (int i = 0; i < dims; i++) {
            if (starts[i] < 0 || starts[i] >= srcShape[i] ||
                    ends[i] <= starts[i] || ends[i] > srcShape[i]) {
                throw new IllegalArgumentException("Invalid slice range at dimension " + i);
            }
        }
        long[] newShape = new long[dims];
        for (int i = 0; i < dims; i++) {
            newShape[i] = ends[i] - starts[i];
            if (newShape[i] <= 0) throw new IllegalArgumentException("Bad slice range");
        }
        OnnxJavaType type = src.getInfo().type;
        int elemSize = type.size;
        ByteBuffer srcBuf = toByteBuffer(src);
        ByteBuffer dstBuf = ByteBuffer.allocateDirect((int) (numElements(newShape) * elemSize))
                .order(ByteOrder.nativeOrder());
        int[] stride = new int[dims];
        stride[dims - 1] = elemSize;
        for (int i = dims - 2; i >= 0; i--) {
            stride[i] = (int) srcShape[i + 1] * stride[i + 1];
        }
        long[] pos = new long[dims];
        while (true) {
            int srcOffset = 0;
            for (int i = 0; i < dims; i++) {
                srcOffset += (int) ((starts[i] + pos[i]) * stride[i]);
            }
            srcBuf.limit(srcOffset + elemSize);
            srcBuf.position(srcOffset);
            dstBuf.put(srcBuf);
            int d = dims - 1;
            while (d >= 0) {
                pos[d]++;
                if (pos[d] < newShape[d]) break;
                pos[d] = 0;
                d--;
            }
            if (d < 0) break;
        }
        dstBuf.rewind();
        return create(dstBuf, newShape, type);
    }


    private static ByteBuffer toByteBuffer(OnnxTensor t) throws OrtException {
        Object arr = t.getValue();
        OnnxJavaType type = t.getInfo().type;
        return toByteBuffer(arr, type);
    }

    private static ByteBuffer toByteBuffer(Object arr) {
        Class<?> arrayClass = arr.getClass();
        while (arrayClass.isArray()) {
            arrayClass = arrayClass.getComponentType();
        }
        return toByteBuffer(arr, autoType(arrayClass));
    }

    private static ByteBuffer toByteBuffer(Object arr, OnnxJavaType type) {
        long[] shape = getArrayShape(arr);
        long totalElements = numElements(shape);
        ByteBuffer buf = ByteBuffer.allocateDirect((int) (totalElements * type.size))
                .order(ByteOrder.nativeOrder());

        // 使用递归方式填充缓冲区
        fillBuffer(arr, buf, type);

        buf.rewind();
        return buf;
    }

    // 递归方法用于填充缓冲区
    private static void fillBuffer(Object array, ByteBuffer buffer, OnnxJavaType type) {
        if (array == null) return;

        if (array.getClass().isArray()) {
            int length = Array.getLength(array);
            for (int i = 0; i < length; i++) {
                fillBuffer(Array.get(array, i), buffer, type);
            }
        } else {
            // 基本数据类型
            switch (type) {
                case FLOAT:
                    buffer.putFloat((Float) array);
                    break;
                case DOUBLE:
                    buffer.putDouble((Double) array);
                    break;
                case INT8:
                case UINT8:
                    buffer.put((Byte) array);
                    break;
                case INT16:
                    buffer.putShort((Short) array);
                    break;
                case INT32:
                    buffer.putInt((Integer) array);
                    break;
                case INT64:
                    buffer.putLong((Long) array);
                    break;
                case BOOL:
                    buffer.put((byte) ((Boolean) array ? 1 : 0));
                    break;
                case FLOAT16:
                case BFLOAT16:
                    buffer.putShort((Short) array);
                    break;
                default:
                    throw new UnsupportedOperationException("Unsupported type: " + type);
            }
        }
    }

    private static long numElements(long[] shape) {
        long n = 1;
        for (long s : shape) {
            n *= s;
            if (n < 0) {
                throw new IllegalArgumentException("Shape too large, would cause overflow");
            }
        }
        return n;
    }

    public OnnxTensor copy(OnnxTensor src) throws OrtException {
        ByteBuffer data = toByteBuffer(src);
        return create(data, src.getInfo().getShape(), src.getInfo().type);
    }

}