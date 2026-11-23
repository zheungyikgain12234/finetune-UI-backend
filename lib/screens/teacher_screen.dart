import 'package:flutter/material.dart';

import '../services/job_runner.dart';
import '../widgets/common.dart';

/// Teacher: label your files with a model or agent that the server offers, then download the labels
/// or hand them to the Student tab. Nothing here names a server path, URL or key.
class TeacherScreen extends StatefulWidget {
  const TeacherScreen({super.key, required this.runner});
  final JobRunner runner;
  @override
  State<TeacherScreen> createState() => _TeacherScreenState();
}

class _TeacherScreenState extends State<TeacherScreen> {
  int mode = 0; // 0 = model (images / PDFs), 1 = agent (text in, JSON out)
  List<Map<String, dynamic>> teachers = [];
  String? teacherId;
  List<Map<String, dynamic>> promptPresets = [];
  String? promptPreset;
  final prompt = TextEditingController();
  final fields = TextEditingController();
  final instructions = TextEditingController();
  final dpi = TextEditingController(text: '200');
  final maxTok = TextEditingController(text: '8192');
  final timeout = TextEditingController(text: '600');
  final secret =
      TextEditingController(); // kept in memory only, sent with the run request
  bool showSecret = false;
  bool markdownPost = true;
  bool thinking = false;
  Map<String, dynamic> ws = {};
  Map<String, dynamic>? testResult;
  bool testing = false;
  String? msg;

  JobRunner get r => widget.runner;
  String get jobName => 'ws:${r.wid}:teacher';
  List<Map<String, dynamic>> get options => [
    for (final t in teachers)
      if (t['kind'] == (mode == 0 ? 'model' : 'agent')) t,
  ];
  Map<String, dynamic>? get teacher {
    for (final t in options) {
      if (t['id'] == teacherId) return t;
    }
    return null;
  }

  @override
  void initState() {
    super.initState();
    _load();
    r.addListener(_onRunner);
  }

  @override
  void dispose() {
    r.removeListener(_onRunner);
    secret.dispose();
    super.dispose();
  }

  void _onRunner() {
    if (teachers.isEmpty && r.error == null) _load();
    _refreshWs();
  }

  Future<void> _load() async {
    try {
      final t = await r.teachers();
      final p = await r.prompts();
      if (!mounted) return;
      setState(() {
        teachers = t;
        promptPresets = p;
        if (teacherId == null && options.isNotEmpty)
          teacherId = options.first['id'];
        if (promptPreset == null && p.isNotEmpty && prompt.text.isEmpty) {
          promptPreset = p.first['name'];
          prompt.text = p.first['text'];
        }
      });
    } catch (_) {}
  }

  Future<void> _refreshWs() async {
    if (r.wid.isEmpty || r.error != null) return;
    try {
      final w = await r.wsStatus();
      if (mounted) setState(() => ws = w);
    } catch (_) {}
  }

  Future<void> test() async {
    if (teacherId == null) return;
    setState(() {
      testing = true;
      testResult = null;
    });
    Map<String, dynamic> res;
    try {
      res = await r.teacherTest(teacherId!);
    } catch (e) {
      res = {'ok': false, 'info': 'Could not reach the backend: $e'};
    }
    if (mounted) {
      setState(() {
        testing = false;
        testResult = res;
      });
    }
  }

  Future<void> run() async {
    if (teacherId == null) {
      setState(() => msg = 'Pick a teacher first.');
      return;
    }
    if (mode == 0 && prompt.text.trim().isEmpty) {
      setState(() => msg = 'Write or pick a prompt first.');
      return;
    }
    if ((ws['slots'] as Map?)?['teacher_inputs'] == null) {
      setState(() => msg = 'Upload the input files (.zip) first.');
      return;
    }
    final t = teacher;
    if (mode == 1 &&
        t != null &&
        t['configured'] == false &&
        secret.text.trim().isEmpty) {
      setState(() => msg = "Enter the agent's secret first.");
      return;
    }
    try {
      await r.teacherRun({
        'teacher': teacherId,
        if (mode == 0) ...{
          'prompt': prompt.text.trim(),
          'dpi': int.tryParse(dpi.text) ?? 200,
          'max_tokens': int.tryParse(maxTok.text) ?? 8192,
          'markdown': markdownPost,
          'thinking': thinking,
        } else ...{
          'fields': fields.text.trim(),
          'instructions': instructions.text.trim(),
          'timeout': int.tryParse(timeout.text) ?? 600,
          if (secret.text.trim().isNotEmpty) 'secret': secret.text.trim(),
        },
      });
      setState(() => msg = 'Started. Logs appear on the right.');
    } catch (e) {
      setState(() => msg = '$e');
    }
  }

  Future<void> useForStudent() async {
    try {
      await r.useTeacherLabels();
      await _refreshWs();
      setState(
        () => msg = 'Done. Open the Student tab: the inputs and labels are already filled in with these files.',
      );
    } catch (e) {
      setState(() => msg = '$e');
    }
  }

  Widget _downloads() {
    final t = ws['teacher'] as Map?;
    if (t == null) return const SizedBox.shrink();
    return Card(
      margin: const EdgeInsets.only(top: 16),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Labels', style: Theme.of(context).textTheme.titleSmall),
            Text(
              '${t['files']} label file(s) written'
              '${(t['flagged'] ?? 0) > 0 ? ', ${t['flagged']} flagged (see flagged.txt in the zip)' : ''}.',
            ),
            gap(8),
            Wrap(
              spacing: 12,
              runSpacing: 8,
              children: [
                FilledButton.icon(
                  onPressed: () => r.download('teacher'),
                  icon: const Icon(Icons.download),
                  label: Text('Download labels (${t['mb']} MB)'),
                ),
                OutlinedButton.icon(
                  onPressed: useForStudent,
                  icon: const Icon(Icons.school_outlined),
                  label: const Text('Use as training data for Student'),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: r,
      builder: (context, _) {
        final job = r.byName(jobName);
        final t = teacher;
        final slot = (ws['slots'] as Map?)?['teacher_inputs'] as Map?;
        final notReady = t != null && t['configured'] == false;
        return Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Expanded(
              flex: 5,
              child: ListView(
                padding: const EdgeInsets.all(16),
                children: [
                  SegmentedButton<int>(
                    segments: const [
                      ButtonSegment(
                        value: 0,
                        icon: Icon(Icons.psychology_alt),
                        label: Text('Model (reads images / PDFs)'),
                      ),
                      ButtonSegment(
                        value: 1,
                        icon: Icon(Icons.hub),
                        label: Text('Agent (reads text, returns JSON)'),
                      ),
                    ],
                    selected: {mode},
                    onSelectionChanged: (v) => setState(() {
                      mode = v.first;
                      teacherId = options.isNotEmpty
                          ? options.first['id']
                          : null;
                      testResult = null;
                      msg = null;
                    }),
                  ),
                  gap(),
                  Section(
                    '1. Teacher',
                    children: [
                      DropdownButtonFormField<String>(
                        key: ValueKey('$mode-${teachers.length}'),
                        initialValue: teacherId,
                        isExpanded: true,
                        decoration: InputDecoration(
                          labelText: mode == 0 ? 'Model' : 'Agent',
                          helperText: teachers.isEmpty
                              ? 'Loading the list from the server…'
                              : (t?['note'] as String?),
                          border: const OutlineInputBorder(),
                        ),
                        items: [
                          for (final o in options)
                            DropdownMenuItem(
                              value: o['id'] as String,
                              child: Text('${o['label']}'),
                            ),
                        ],
                        onChanged: (v) => setState(() {
                          teacherId = v;
                          testResult = null;
                        }),
                      ),
                      if (mode == 1 && t != null) ...[
                        const SizedBox(height: 12),
                        TextField(
                          controller: secret,
                          obscureText: !showSecret,
                          enableSuggestions: false,
                          autocorrect: false,
                          onChanged: (_) => setState(() {}),
                          decoration: InputDecoration(
                            labelText: notReady
                                ? 'Agent secret (required)'
                                : 'Agent secret (optional)',
                            helperText: notReady
                                ? 'The server has no secret for this agent. Enter yours; it is used for this run only and is not saved.'
                                : 'Leave empty to use the server\'s secret, or enter your own for this run only.',
                            helperMaxLines: 2,
                            border: const OutlineInputBorder(),
                            suffixIcon: IconButton(
                              icon: Icon(
                                showSecret
                                    ? Icons.visibility_off
                                    : Icons.visibility,
                              ),
                              onPressed: () =>
                                  setState(() => showSecret = !showSecret),
                            ),
                          ),
                        ),
                      ],
                      gap(),
                      Row(
                        children: [
                          FilledButton.icon(
                            onPressed: (testing || teacherId == null)
                                ? null
                                : test,
                            icon: testing
                                ? const SizedBox(
                                    width: 16,
                                    height: 16,
                                    child: CircularProgressIndicator(
                                      strokeWidth: 2,
                                    ),
                                  )
                                : const Icon(Icons.wifi_tethering),
                            label: Text(
                              mode == 0
                                  ? 'Test connection (hello world)'
                                  : 'Test agent (health + card)',
                            ),
                          ),
                          gap(),
                          if (testResult != null)
                            Expanded(
                              child: Text(
                                '${testResult!['ok'] == true ? '✓' : '✗'} ${testResult!['info'] ?? testResult!['error']}',
                                style: TextStyle(
                                  color: testResult!['ok'] == true
                                      ? Colors.green
                                      : Colors.redAccent,
                                ),
                              ),
                            ),
                        ],
                      ),
                    ],
                  ),
                  if (mode == 0)
                    Section(
                      '2. Prompt',
                      children: [
                        DropdownButtonFormField<String>(
                          key: ValueKey(promptPresets.length),
                          initialValue: promptPreset,
                          decoration: const InputDecoration(
                            labelText: 'Preset (editable below)',
                            border: OutlineInputBorder(),
                          ),
                          items: [
                            for (final p in promptPresets)
                              DropdownMenuItem(
                                value: p['name'] as String,
                                child: Text(p['name']),
                              ),
                          ],
                          onChanged: (n) => setState(() {
                            promptPreset = n;
                            prompt.text = promptPresets.firstWhere(
                              (p) => p['name'] == n,
                            )['text'];
                          }),
                        ),
                        gap(),
                        TextField(
                          controller: prompt,
                          minLines: 6,
                          maxLines: 14,
                          decoration: const InputDecoration(
                            labelText: 'Prompt sent with every image / page',
                            alignLabelWithHint: true,
                            border: OutlineInputBorder(),
                          ),
                        ),
                      ],
                    )
                  else
                    Section(
                      '2. What to save',
                      children: [
                        TextField(
                          controller: fields,
                          minLines: 3,
                          maxLines: 8,
                          decoration: const InputDecoration(
                            labelText: 'Parts to keep (comma or one per line). Empty = save the FULL agent response',
                            helperText:
                                'Paths start at the top of the saved response, e.g.  response.outputs.structured  '
                                '(only the extracted JSON).  A bare name like  document_number, line_items.description  '
                                'is searched inside it and also sent to the agent as what to extract.',
                            helperMaxLines: 4,
                            alignLabelWithHint: true,
                            border: OutlineInputBorder(),
                          ),
                        ),
                        gap(),
                        TextField(
                          controller: instructions,
                          minLines: 2,
                          maxLines: 5,
                          decoration: const InputDecoration(
                            labelText:
                                'Extra instructions for the agent (optional)',
                            alignLabelWithHint: true,
                            border: OutlineInputBorder(),
                          ),
                        ),
                      ],
                    ),
                  Section(
                    '3. Files',
                    children: [
                      ZipUploadTile(
                        runner: r,
                        slot: 'teacher_inputs',
                        title: 'Input files (.zip)',
                        hint: mode == 0
                            ? 'PDFs or images (searched recursively).'
                            : 'Markdown / text files (.md), searched recursively.',
                        info: slot,
                        onDone: _refreshWs,
                      ),
                      ExpansionTile(
                        tilePadding: EdgeInsets.zero,
                        title: const Text('Advanced'),
                        childrenPadding: const EdgeInsets.only(bottom: 8),
                        children: [
                          if (mode == 0) ...[
                            Row(
                              children: [
                                Expanded(
                                  child: TextField(
                                    controller: dpi,
                                    decoration: const InputDecoration(
                                      labelText: 'PDF render DPI',
                                      border: OutlineInputBorder(),
                                    ),
                                  ),
                                ),
                                gap(),
                                Expanded(
                                  child: TextField(
                                    controller: maxTok,
                                    decoration: const InputDecoration(
                                      labelText: 'Max tokens',
                                      border: OutlineInputBorder(),
                                    ),
                                  ),
                                ),
                              ],
                            ),
                            CheckboxListTile(
                              dense: true,
                              contentPadding: EdgeInsets.zero,
                              value: !thinking,
                              onChanged: (v) =>
                                  setState(() => thinking = !(v ?? true)),
                              title: const Text(
                                'Disable thinking (sends enable_thinking=false; vLLM/Qwen3 only)',
                              ),
                            ),
                            CheckboxListTile(
                              dense: true,
                              contentPadding: EdgeInsets.zero,
                              value: markdownPost,
                              onChanged: (v) =>
                                  setState(() => markdownPost = v ?? true),
                              title: const Text(
                                'Markdown clean-up (merge table rows, strip links, add page markers)',
                              ),
                              subtitle: const Text(
                                'Turn off for non-document tasks to keep the raw model text.',
                              ),
                            ),
                          ] else
                            TextField(
                              controller: timeout,
                              decoration: const InputDecoration(
                                labelText: 'Timeout per file (s)',
                                border: OutlineInputBorder(),
                              ),
                            ),
                        ],
                      ),
                      gap(),
                      Row(
                        children: [
                          FilledButton.icon(
                            onPressed:
                                (job != null && job.running) ||
                                    (notReady && !(mode == 1 && secret.text.trim().isNotEmpty))
                                ? null
                                : run,
                            icon: const Icon(Icons.play_arrow),
                            label: const Text('Run teacher in background'),
                          ),
                          gap(),
                          if (job != null && job.running)
                            OutlinedButton.icon(
                              onPressed: () => r.stop(job),
                              icon: const Icon(
                                Icons.stop_circle_outlined,
                                color: Colors.redAccent,
                              ),
                              label: const Text('Stop'),
                            ),
                          gap(),
                          if (msg != null) Expanded(child: Text(msg!)),
                        ],
                      ),
                      const SizedBox(height: 4),
                      const Text(
                        'Each run starts fresh: the previous labels in your workspace are replaced.',
                        style: TextStyle(fontSize: 12),
                      ),
                      _downloads(),
                    ],
                  ),
                ],
              ),
            ),
            Expanded(
              flex: 4,
              child: Padding(
                padding: const EdgeInsets.fromLTRB(0, 16, 16, 16),
                child: job == null
                    ? const Center(
                        child: Text(
                          'No teacher job yet. Logs will appear here.',
                        ),
                      )
                    : LogViewer(runner: r, job: job),
              ),
            ),
          ],
        );
      },
    );
  }
}
