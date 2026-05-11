import 'package:flutter_test/flutter_test.dart';
import 'package:tushu/services/excel_service.dart';

void main() {
  test('append rows are aligned to existing headers', () {
    final headers = ExcelService.headersForAppend(
      ['姓名', '电话'],
      ['电话', '姓名'],
    );

    expect(headers, ['姓名', '电话']);
    expect(
      ExcelService.rowForHeaders({'电话': '13800138000', '姓名': '张三'}, headers),
      ['张三', '13800138000'],
    );
  });

  test('append headers are extended for new fields without dropping data', () {
    final headers = ExcelService.headersForAppend(
      ['姓名', '电话'],
      ['电话', '公司', '姓名'],
    );

    expect(headers, ['姓名', '电话', '公司']);
    expect(
      ExcelService.rowForHeaders(
        {'电话': '13800138000', '公司': '示例公司', '姓名': '张三'},
        headers,
      ),
      ['张三', '13800138000', '示例公司'],
    );
  });
}
