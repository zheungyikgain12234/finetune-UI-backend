import 'dart:async';
import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;

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

/// HTTP client for scripts/server.py, which runs the real jobs.
class JobRunner extends ChangeNotifier {
  JobRunner({this.base = 'http://127.0.0.1:8765'}) {
    refresh();
    _timer = Timer.periodic(const Duration(seconds: 3), (_) => refresh());
  }

  final String base;
  String root = '';
  String? error;
  List<Job> jobs = [];
  Timer? _timer;

  Future<void> refresh() async {
    try {
      if (root.isEmpty)
        root = jsonDecode(
          (await http.get(Uri.parse('$base/api/info'))).body,
        )['root'];
      final r = await http.get(Uri.parse('$base/api/jobs'));
      jobs = [for (final j in jsonDecode(r.body)) Job(j)];
      error = null;
    } catch (e) {
      error =
          'Backend not reachable at $base. Start it with: python3 ovis_ft_ui/scripts/server.py';
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
    final r = await http.post(
      Uri.parse('$base/api/jobs'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'name': name, 'cmd': cmd, 'env': env}),
    );
    final job = Job(jsonDecode(r.body));
    await refresh();
    return job;
  }

  Future<void> stop(Job job) async {
    await http.post(Uri.parse('$base/api/jobs/${job.id}/stop'), body: '{}');
    await refresh();
  }

  Future<String> tail(Job job, {int maxBytes = 60000}) async {
    try {
      final r = await http.get(
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
      final r = await http.get(
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
    final r = await http.get(Uri.parse('$base/api/prompts'));
    return List<Map<String, dynamic>>.from(jsonDecode(r.body));
  }

  /// Training recipes (scripts/recipes/*.json): student model, hyper-parameters and command templates.
  Future<List<Map<String, dynamic>>> recipes() async {
    final r = await http.get(Uri.parse('$base/api/recipes'));
    return List<Map<String, dynamic>>.from(jsonDecode(r.body));
  }

  Future<Map<String, dynamic>> ls(String path) async {
    final r = await http.get(
      Uri.parse('$base/api/ls').replace(queryParameters: {'path': path.trim()}),
    );
    return jsonDecode(r.body);
  }

  Future<void> mkdir(String path) async {
    final r = await http.post(
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
    final r = await http.post(
      Uri.parse('$base/api/test-connection'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'url': url, 'key': key, 'model': model}),
    );
    return jsonDecode(r.body);
  }

  /// Checks a Hugging Face model id (or local folder). Returns {ok, info|error}.
  Future<Map<String, dynamic>> testHfModel(String model) async {
    final r = await http.post(
      Uri.parse('$base/api/test-hf-model'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'model': model}),
    );
    return jsonDecode(r.body);
  }

  /// Health check + agent card (name, capability, input fields) of a BYOA/ADK agent.
  Future<Map<String, dynamic>> agentCard(String url) async {
    final r = await http.post(
      Uri.parse('$base/api/agent-card'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'url': url}),
    );
    return jsonDecode(r.body);
  }

  Future<List<String>> listModels(String url, String key) async {
    final r = await http.post(
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
