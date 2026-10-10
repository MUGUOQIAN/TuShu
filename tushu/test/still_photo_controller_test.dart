import 'package:camera/camera.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:tushu/pages/camera_page.dart';

void main() {
  test('静帧相机不采集音频，避免麦克风权限阻断拍照', () {
    final controller = createStillPhotoController(
      const CameraDescription(
        name: 'back',
        lensDirection: CameraLensDirection.back,
        sensorOrientation: 90,
      ),
    );
    addTearDown(controller.dispose);

    expect(controller.enableAudio, isFalse);
  });
}
