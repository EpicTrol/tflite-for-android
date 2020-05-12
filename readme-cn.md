# 图像识别检测流程

1. onCreate中请求相机权限并设置页面内容区的fragment
   在`CameraActivity.java`的onCreate()设置window layout，以及设置contentView，请求打开相机的权限`requestPermission()`，然后进行设置相机实时图片预览区域的Fragment: `setFragment()`
   
   `CameraConnectionFragment.java`:
   
   构造fragment时传入了两个比较重要的回调，一个是在打开摄像头时回调 `ConnectionCallback`,另一个是在摄像头拍摄到图片时回调`OnImageAvailableListener`
   fragment的生命周期中的几个重要方法。onCreateView() onViewCreated()基本没做太多事情，onResume()中有个关键动作，它调用了openCamera()方法来打开摄像头：
   
   ```
   public void onResume() {
     super.onResume();
     startBackgroundThread();
     
     if (textureView.isAvailable()) {
       // 屏幕没有处于关闭状态时，打开摄像头。textureView是fragment中展示摄像头实时捕获的图片的区域。
       openCamera(textureView.getWidth(), textureView.getHeight());
     } else {
       textureView.setSurfaceTextureListener(surfaceTextureListener);
     }
   }
   ```
   
   
   
2. 打开摄像头，并注册`ConnectionCallback`和`OnImageAvailableListener`

    + `ConnectionCallback`回调流程：

      在`openCamera()`中的`setUpCameraOutputs()`设置camera捕获图片的一些参数。如图片预览大小previewSize，摄像头方向sensorOrientation等。最重要的是回调我们之前传入到fragment中的`ConnectionCallback`的`onPreviewSizeChosen()`方法。

    ```
    new CameraConnectionFragment.ConnectionCallback() {
     @Override
     // 选择了预览图片的大小 回调
     public void onPreviewSizeChosen(final Size size, final int rotation) {
      // 获取相机捕获的图片的宽高，以及相机旋转方向。
      previewHeight = size.getHeight();
      previewWidth = size.getWidth();
      // 相机捕获的图片的大小确定后，需要对捕获图片做裁剪等预操作。这将回调到DetectorActivity中
      CameraActivity.this.onPreviewSizeChosen(size, rotation);
      }
    }
    ```

    + `OnImageAvailableListener`回调流程：

      在摄像头被打开后，捕获的图片available时由系统回调到。摄像头打开后，会create一个新的预览session，其中就会设置`OnImageAvailableListener`到CameraDevice中。

      

3. 相机预览图片宽高确定后，在`DetectorActivity`回调`onPreviewSizeChosen()`,主要是构造检测器detector

   ```
   // 构造检测器，利用了TensorFlow训练出来的Model,包括model,标签,输入size,是否量化模型quantize
   try {
     detector =
         TFLiteObjectDetectionAPIModel.create(
             getAssets(),
             TF_OD_API_MODEL_FILE,
             TF_OD_API_LABELS_FILE,
             TF_OD_API_INPUT_SIZE,
             TF_OD_API_IS_QUANTIZED);
     cropSize = TF_OD_API_INPUT_SIZE;
   } 
   ```

   

   + 预处理预览图片`ImageUtil.java`

   `public static Matrix getTransformationMatrix`

    预处理预览图片，裁剪，旋转等操作。
   srcWidth, srcHeight为预览图片宽高。dstWidth dstHeight为训练模型时使用的图片的宽高
   applyRotation 为旋转角度，必须是90的倍数，maintainAspectRatio 如果为true，旋转时缩放x而保证y不变

   

   +  相机预览图片available时，`OnImageAvailableListener`回调(Camera2 API)

    当相机预览图片准备好时，Android系统的cameraDevice会回调之前注册的`OnImageAvailableListener`。下面是OnImageAvailableListener`做的事情：

   先做一些预校验，如previewWidth是否被设置，当前是否正在处理图片等。

   然后将相机捕获的yuv格式图像转为rgb格式。

   最后也是最重要的一步，调用`processImage()`，利用TensorFlow模型来处理图片（一知半解）。

   ```
   @Override
   public void onImageAvailable(final ImageReader reader) {
     // onPreviewSizeChosen被回调后，设置了previewWidth和previewHeight，才处理预览图片
     if (previewWidth == 0 || previewHeight == 0) {
       return;
     }
     // 构造图片输出矩阵
     if (rgbBytes == null) {
       rgbBytes = new int[previewWidth * previewHeight];
     }
     try {
       // 获取图片
       final Image image = reader.acquireLatestImage();
   
       if (image == null) {
         return;
       }
   
       // 正在处理图片时，则直接返回
       if (isProcessingFrame) {
         image.close();
         return;
       }
       // yuv转换为rgb格式
       isProcessingFrame = true;
       Trace.beginSection("imageAvailable");
       final Plane[] planes = image.getPlanes();
       fillBytes(planes, yuvBytes);
       yRowStride = planes[0].getRowStride();
       final int uvRowStride = planes[1].getRowStride();
       final int uvPixelStride = planes[1].getPixelStride();
   
       imageConverter =
           new Runnable() {
             @Override
             public void run() {
               ImageUtils.convertYUV420ToARGB8888(
                   yuvBytes[0],
                   yuvBytes[1],
                   yuvBytes[2],
                   previewWidth,
                   previewHeight,
                   yRowStride,
                   uvRowStride,
                   uvPixelStride,
                   rgbBytes);
             }
           };
   
       postInferenceCallback =
           new Runnable() {
             @Override
             public void run() {
               image.close();
               isProcessingFrame = false;
             }
           };
   
       // 利用训练模型来预测图片
       processImage();
     } catch (final Exception e) {
       LOGGER.e(e, "Exception!");
       Trace.endSection();
       return;
     }
     Trace.endSection();
   }
   ```

   粗略观察分析一下`processImage()`：(不懂)

   代码位于`DetectorActivity`中，先绘制图片，然后在`runInBackground`中

   用classifier对图片进行识别，得到输入图片为每个分类的概率？

   ```
   final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);
   ```

   其中`recognizeImage`方法在`TFLiteObjectDetectionAPIModel.java`中，过程大概如下：
   
      1. 预处理输入图片，读取像素点，并将RGB三通道数值归一化
   
      2. 将输入数据拷贝到TensorFlow中，并feed数据给模型
   
      3. 跑模型run
   
      4. 将数据整合为Recognition对象return
   
         

### 总结

打开摄像头——>注册监听器——>构造分类器classifier？——>预处理相机图片——>利用模型预测图片识别检测



### 使用

根据模型的不同更换asset文件夹中的.tflite文件和labelmap标签文件，同时根据模型的不同视情况修改源代码

由于模型还未完全调试好所以.tflite文件暂不上传，但初步的apk测试文件已完成