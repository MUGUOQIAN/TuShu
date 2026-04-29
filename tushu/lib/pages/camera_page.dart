import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:image_picker/image_picker.dart';
import '../services/api_service.dart';
import '../widgets/template_selector.dart';
import '../models/template.dart';
import 'result_page.dart';

class CameraPage extends StatefulWidget {
  const CameraPage({super.key});

  @override
  State<CameraPage> createState() => _CameraPageState();
}

class _CameraPageState extends State<CameraPage> {
  CameraController? _controller;
  List<CameraDescription> _cameras = [];
  TemplateType _selectedTemplate = TemplateType.custom;
  String _customFields = "";
  bool _isProcessing = false;

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _initCamera() async {
    _cameras = await availableCameras();
    if (_cameras.isNotEmpty) {
      _controller = CameraController(_cameras[0], ResolutionPreset.medium);
      await _controller!.initialize();
      setState(() {});
    }
  }

  /// 拍照
  Future<void> _takePicture() async {
    if (_controller == null || _isProcessing) return;
    try {
      final XFile photo = await _controller!.takePicture();
      _processImage(File(photo.path));
    } catch (e) {
      _showError("拍照失败: $e");
    }
  }

  /// 从相册选择
  Future<void> _pickFromGallery() async {
    final picker = ImagePicker();
    final XFile? image = await picker.pickImage(
      source: ImageSource.gallery,
      imageQuality: 80, // 预压缩
    );
    if (image != null) {
      _processImage(File(image.path));
    }
  }

  /// 处理图片：压缩→调API→跳转结果页
  Future<void> _processImage(File imageFile) async {
    setState(() => _isProcessing = true);

    try {
      // 1. 读取并压缩为Base64
      final bytes = await imageFile.readAsBytes();
      final base64Image = base64Encode(bytes);

      // 2. 调用后端API
      final result = await ApiService.recognize(
        imageBase64: base64Image,
        templateType: _selectedTemplate,
        customFields: _customFields,
      );

      // 3. 跳转结果页
      if (mounted) {
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (_) => ResultPage(
              imagePath: imageFile.path,
              extractedData: result,
              templateType: _selectedTemplate,
            ),
          ),
        );
      }
    } catch (e) {
      _showError("识别失败，请重试");
    } finally {
      setState(() => _isProcessing = false);
    }
  }

  void _showError(String msg) {
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          // 相机预览
          if (_controller?.value.isInitialized == true)
            Positioned.fill(child: CameraPreview(_controller!)),

          // 顶部：相册导入按钮
          Positioned(
            top: 50, right: 20,
            child: IconButton(
              icon: const Icon(Icons.photo_library, color: Colors.white, size: 32),
              onPressed: _pickFromGallery,
            ),
          ),

          // 底部：模板选择器
          Positioned(
            bottom: 140, left: 0, right: 0,
            child: TemplateSelector(
              selected: _selectedTemplate,
              customFields: _customFields,
              onTemplateChanged: (template, fields) {
                setState(() {
                  _selectedTemplate = template;
                  _customFields = fields;
                });
              },
            ),
          ),

          // 底部：快门按钮
          Positioned(
            bottom: 40, left: 0, right: 0,
            child: Center(
              child: GestureDetector(
                onTap: _isProcessing ? null : _takePicture,
                child: Container(
                  width: 72, height: 72,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    border: Border.all(color: Colors.white, width: 4),
                    color: _isProcessing ? Colors.grey : Colors.transparent,
                  ),
                  child: _isProcessing
                      ? const CircularProgressIndicator(color: Colors.white)
                      : null,
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }
}