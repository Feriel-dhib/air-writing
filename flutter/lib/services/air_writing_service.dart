import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;

// ---------------------------------------------------------------------------
// Data classes
// ---------------------------------------------------------------------------

class CharScore {
  final String char;
  final double confidence;

  const CharScore({required this.char, required this.confidence});

  factory CharScore.fromJson(Map<String, dynamic> json) => CharScore(
        char: json['char'] as String,
        confidence: (json['confidence'] as num).toDouble(),
      );

  @override
  String toString() => '$char (${(confidence * 100).toStringAsFixed(1)}%)';
}

class PredictionResult {
  final String char;
  final double confidence;
  final List<CharScore> top3;

  const PredictionResult({
    required this.char,
    required this.confidence,
    required this.top3,
  });

  factory PredictionResult.fromJson(Map<String, dynamic> json) =>
      PredictionResult(
        char: json['char'] as String,
        confidence: (json['confidence'] as num).toDouble(),
        top3: (json['top3'] as List<dynamic>)
            .map((e) => CharScore.fromJson(e as Map<String, dynamic>))
            .toList(),
      );

  @override
  String toString() => 'PredictionResult($char, ${(confidence * 100).toStringAsFixed(1)}%)';
}

// ---------------------------------------------------------------------------
// Exceptions
// ---------------------------------------------------------------------------

class AirWritingException implements Exception {
  final String message;
  final int? statusCode;
  final String? body;

  const AirWritingException(this.message, {this.statusCode, this.body});

  @override
  String toString() => 'AirWritingException: $message (HTTP $statusCode)';
}

// ---------------------------------------------------------------------------
// Service
// ---------------------------------------------------------------------------

/// Stateless HTTP client for the Air Writing FastAPI server.
///
/// Usage:
/// ```dart
/// final service = AirWritingService(
///   baseUrl: 'https://air-writing-api.onrender.com',
/// );
///
/// // Wake the server early (Render free tier sleeps after inactivity)
/// await service.warmUp();
///
/// // Send a JPEG frame for prediction
/// final result = await service.predict(jpegBytes);
/// print('${result.char} — ${result.confidence}');
/// ```
class AirWritingService {
  final String baseUrl;

  /// Timeout for normal requests (server already awake).
  final Duration timeout;

  /// Timeout for the first request / cold start (Render wake-up ≤ 30 s).
  final Duration coldStartTimeout;

  bool _serverAwake = false;

  AirWritingService({
    required this.baseUrl,
    this.timeout = const Duration(seconds: 10),
    this.coldStartTimeout = const Duration(seconds: 60),
  });

  /// Whether the server has responded at least once in this session.
  bool get isServerAwake => _serverAwake;

  // -----------------------------------------------------------------------
  // Health
  // -----------------------------------------------------------------------

  /// Lightweight liveness check.  Returns `true` if the server is healthy.
  Future<bool> checkHealth() async {
    try {
      final response = await http
          .get(Uri.parse('$baseUrl/health'))
          .timeout(timeout);
      if (response.statusCode == 200) {
        _serverAwake = true;
        return true;
      }
      return false;
    } catch (_) {
      return false;
    }
  }

  /// Wake the server up (call early, e.g. on app launch, to hide cold-start
  /// latency).  Returns `true` once the server responds.
  Future<bool> warmUp() async {
    try {
      final response = await http
          .get(Uri.parse('$baseUrl/health'))
          .timeout(coldStartTimeout);
      _serverAwake = response.statusCode == 200;
      return _serverAwake;
    } catch (_) {
      _serverAwake = false;
      return false;
    }
  }

  // -----------------------------------------------------------------------
  // Prediction
  // -----------------------------------------------------------------------

  /// Send a JPEG image to the server and return the predicted character.
  ///
  /// [jpegBytes] — raw JPEG bytes (e.g. from `controller.takePicture()`
  ///   or a rendered 28×28 canvas).
  ///
  /// [alreadyRendered] — set to `true` if [jpegBytes] is already a 28×28
  ///   grayscale image ready for the CNN.  When `false` (default) the server
  ///   crops, resizes and re-centers the stroke automatically.
  ///
  /// Automatically retries once on timeout (handles Render cold start).
  Future<PredictionResult> predict(
    Uint8List jpegBytes, {
    bool alreadyRendered = false,
  }) async {
    final payload = jsonEncode({
      'image_b64': base64Encode(jpegBytes),
      'already_rendered': alreadyRendered,
    });

    final uri = Uri.parse('$baseUrl/predict');
    final headers = {'Content-Type': 'application/json'};
    final effectiveTimeout = _serverAwake ? timeout : coldStartTimeout;

    http.Response response;
    try {
      response = await http
          .post(uri, headers: headers, body: payload)
          .timeout(effectiveTimeout);
    } on TimeoutException {
      // Retry once — server may have been sleeping
      _serverAwake = false;
      try {
        response = await http
            .post(uri, headers: headers, body: payload)
            .timeout(coldStartTimeout);
      } on TimeoutException {
        throw const AirWritingException(
          'Server unreachable after retry (timeout)',
        );
      }
    }

    if (response.statusCode == 503) {
      throw const AirWritingException(
        'Model not loaded yet — server is still starting',
        statusCode: 503,
      );
    }

    if (response.statusCode != 200) {
      throw AirWritingException(
        'Prediction failed',
        statusCode: response.statusCode,
        body: response.body,
      );
    }

    _serverAwake = true;
    final json = jsonDecode(response.body) as Map<String, dynamic>;
    return PredictionResult.fromJson(json);
  }
}
