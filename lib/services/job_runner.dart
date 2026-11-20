import 'dart:async';
import 'dart:convert';
import 'dart:math';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import 'package:url_launcher/url_launcher.dart';

class Job {
  Job(this.j);
  final Map<String, dynamic> j;
  String get id => j['id'];
  String get name => j['name'];
  String get cmd => j['cmd'];
  bool get running => j['running'] == true;
  String get status => j['status'];
  DateTime get startedAt => DateTime.parse(j['startedAt']);
}

class Fold {
  Fold(this.j);
  final Map<String, dynamic> j;
  String get path => j['path'];
  String get name => j['name'];
  String get status => j['predicted'] == true && j['trained'] == true
      ? 'done'
      : j['trained'] == true
      ? 'trained'
      : j['hasData'] == true
      ? 'ready'
      : 'no data';
}

/// Adds the workspace id to every request so the server only shows this browser its own jobs.
class _WsClient extends http.BaseClient {
  _WsClient(this.wid);
  final String Function() wid;
  final _inner = http.Client();
  @override
  Future<http.StreamedResponse> send(http.BaseRequest request) {
    if (wid().isNotEmpty) request.headers['X-Workspace'] = wid();
    return _inner.send(request);
  }
}

/// Backend address. Set at build time for a deployed site:
///   flutter build web --dart-define=API_BASE=https://ft-api.example.com
const kApiBase = String.fromEnvironment('API_BASE', defaultValue: 'http://127.0.0.1:8765');

/// HTTP client for scripts/server.py, which runs the real jobs.
class JobRunner extends ChangeNotifier {
  JobRunner({this.base = kApiBase}) {
    _c = _WsClient(() => wid);
    _init();
    _timer = Timer.periodic(const Duration(seconds: 3), (_) => refresh());
  }

  final String base;
  late final http.Client _c;

  /// Random id of this browser's workspace (uploads, dataset and trained model live under it on the server).
  /// Kept in local storage so a reload does not lose the trained model.
  String wid = '';

  Future<void> _init() async {
    try {
      final sp = await SharedPreferences.getInstance();
      wid = sp.getString('workspace') ?? '';
      if (!RegExp(r'^[a-f0-9]{32}$').hasMatch(wid)) {
        final rnd = Random.secure();
        wid = List.generate(32, (_) => rnd.nextInt(16).toRadixString(16)).join();
        await sp.setString('workspace', wid);
      }
    } catch (_) {
      final rnd = Random.secure();
      wid = List.generate(32, (_) => rnd.nextInt(16).toRadixString(16)).join();
    }
    await refresh();
  }

  // ---- Student quick start: server-side workspace (no folder paths) ----

  /// Models the server allows: [{id, label, vision}].
  Future<List<Map<String, dynamic>>> allowedModels() async {
    final r = await _c.get(Uri.parse('$base/api/allowed-models'));
    return List<Map<String, dynamic>>.from(jsonDecode(r.body));
  }

  /// Uploads a .zip into slot inputs | labels | test_inputs | test_labels. Returns {files, skipped, types}.
  Future<Map<String, dynamic>> uploadZip(String slot, Uint8List bytes) async {
    final r = await _c.post(
      Uri.parse('$base/api/ws/$wid/upload/$slot'),
      headers: {'Content-Type': 'application/zip'},
      body: bytes,
    );
    final b = jsonDecode(r.body);
    if (r.statusCode != 200) throw b['error'] ?? r.body;
    return b;
  }

  Future<Map<String, dynamic>> wsStatus() async {
    final r = await _c.get(Uri.parse('$base/api/ws/$wid/status'));
    return jsonDecode(r.body);
  }

  /// Runs one step of quick_finetune on the server. `p` has: step, model, prompt, holdout, epochs, lr, rank, max_len, max_new.
  Future<void> wsRun(Map<String, dynamic> p) async {
    final r = await _c.post(
      Uri.parse('$base/api/ws/$wid/run'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode(p),
    );
    final b = jsonDecode(r.body);
    if (r.statusCode != 200) throw b['error'] ?? r.body;
    await refresh();
  }

  // ---- Teacher: allowlisted models/agents on the server (no URLs or keys in the browser) ----

  /// [{id, kind: model|agent, label, vision, configured, note}]
  Future<List<Map<String, dynamic>>> teachers() async {
    final r = await _c.get(Uri.parse('$base/api/teachers'));
    return List<Map<String, dynamic>>.from(jsonDecode(r.body));
  }

  Future<Map<String, dynamic>> teacherTest(String id) async {
    final r = await _c.post(
      Uri.parse('$base/api/ws/$wid/teacher/test'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'teacher': id}),
    );
    return jsonDecode(r.body);
  }

  /// `p`: teacher, and for a model: prompt, dpi, max_tokens, markdown, thinking; for an agent: fields, instructions, timeout.
  Future<void> teacherRun(Map<String, dynamic> p) async {
    final r = await _c.post(
      Uri.parse('$base/api/ws/$wid/teacher/run'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode(p),
    );
    final b = jsonDecode(r.body);
    if (r.statusCode != 200) throw b['error'] ?? r.body;
    await refresh();
  }

  /// Copies the teacher's inputs and labels into the Student's inputs and labels.
  Future<void> useTeacherLabels() async {
    final r = await _c.post(Uri.parse('$base/api/ws/$wid/teacher/use-labels'), body: '{}');
    final b = jsonDecode(r.body);
    if (r.statusCode != 200) throw b['error'] ?? r.body;
  }

  /// Opens the download in the browser: what = 'lora' (small adapter), 'merged' (full model, about 16 GB) or 'teacher' (label files).
  Future<bool> download(String what) =>
      launchUrl(Uri.parse('$base/api/ws/$wid/download?what=$what'), mode: LaunchMode.externalApplication);

  String root = '';
  String? error;
  List<Job> jobs = [];
  Timer? _timer;

  Future<void> refresh() async {
    try {
      if (root.isEmpty)
        root = jsonDecode(
          (await _c.get(Uri.parse('$base/api/info'))).body,
        )['root'];
      final r = await _c.get(Uri.parse('$base/api/jobs'));
      jobs = [for (final j in jsonDecode(r.body)) Job(j)];
      error = null;
    } catch (e) {
      error =
          'Backend not reachable at $base. Start it with: python3 ovis_ft_ui/scripts/server.py (or check API_BASE).';
    }
    notifyListeners();
  }

  Job? byName(String name) {
    for (final j in jobs.reversed) {
      if (j.name == name) return j;
    }
    return null;
  }

  bool isRunning(String name) => byName(name)?.running ?? false;

  Future<Job> start(
    String name,
    String cmd, {
    Map<String, String> env = const {},
  }) async {
    final r = await _c.post(
      Uri.parse('$base/api/jobs'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'name': name, 'cmd': cmd, 'env': env}),
    );
    final job = Job(jsonDecode(r.body));
    await refresh();
    return job;
  }

  Future<void> stop(Job job) async {
    await _c.post(Uri.parse('$base/api/jobs/${job.id}/stop'), body: '{}');
    await refresh();
  }

  Future<String> tail(Job job, {int maxBytes = 60000}) async {
    try {
      final r = await _c.get(
        Uri.parse('$base/api/jobs/${job.id}/log?bytes=$maxBytes'),
      );
      return utf8.decode(r.bodyBytes, allowMalformed: true);
    } catch (e) {
      return '(cannot read log: $e)';
    }
  }

  Future<List<Fold>> folds(String path, {String marker = 'outputs'}) async {
    if (path.trim().isEmpty) return [];
    try {
      final r = await _c.get(
        Uri.parse('$base/api/folds')
            .replace(queryParameters: {'path': path.trim(), 'marker': marker}),
      );
      return [for (final f in jsonDecode(r.body)) Fold(f)];
    } catch (_) {
      return [];
    }
  }

  /// Prompt presets (scripts/prompts/*.txt) as [{name, text}].
  Future<List<Map<String, dynamic>>> prompts() async {
    final r = await _c.get(Uri.parse('$base/api/prompts'));
    return List<Map<String, dynamic>>.from(jsonDecode(r.body));
  }

  /// Training recipes (scripts/recipes/*.json): student model, hyper-parameters and command templates.
  Future<List<Map<String, dynamic>>> recipes() async {
    final r = await _c.get(Uri.parse('$base/api/recipes'));
    return List<Map<String, dynamic>>.from(jsonDecode(r.body));
  }

  Future<Map<String, dynamic>> ls(String path) async {
    final r = await _c.get(
      Uri.parse('$base/api/ls').replace(queryParameters: {'path': path.trim()}),
    );
    return jsonDecode(r.body);
  }

  Future<void> mkdir(String path) async {
    final r = await _c.post(
      Uri.parse('$base/api/mkdir'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'path': path}),
    );
    if (r.statusCode != 200) throw jsonDecode(r.body)['error'];
  }

  /// Sends "hello world" through the teacher's chat endpoint. Returns {ok, reply|error, ms}.
  Future<Map<String, dynamic>> testConnection(
    String url,
    String key,
    String model,
  ) async {
    final r = await _c.post(
      Uri.parse('$base/api/test-connection'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'url': url, 'key': key, 'model': model}),
    );
    return jsonDecode(r.body);
  }

  /// Checks a Hugging Face model id (or local folder). Returns {ok, info|error}.
  Future<Map<String, dynamic>> testHfModel(String model) async {
    final r = await _c.post(
      Uri.parse('$base/api/test-hf-model'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'model': model}),
    );
    return jsonDecode(r.body);
  }

  /// Health check + agent card (name, capability, input fields) of a BYOA/ADK agent.
  Future<Map<String, dynamic>> agentCard(String url) async {
    final r = await _c.post(
      Uri.parse('$base/api/agent-card'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'url': url}),
    );
    return jsonDecode(r.body);
  }

  Future<List<String>> listModels(String url, String key) async {
    final r = await _c.post(
      Uri.parse('$base/api/models'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'url': url, 'key': key}),
    );
    final b = jsonDecode(r.body);
    if (r.statusCode != 200) throw b['error'] ?? r.body;
    return List<String>.from(b['models']);
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }
}

String shq(String s) => "'${s.replaceAll("'", "'\\''")}'";
