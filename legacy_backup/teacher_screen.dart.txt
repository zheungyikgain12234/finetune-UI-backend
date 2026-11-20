import 'package:flutter/material.dart';

import '../services/job_runner.dart';
import '../widgets/common.dart';
import 'agent_form.dart';

class _Preset {
  const _Preset(this.name, this.url, this.needsKey);
  final String name, url;
  final bool needsKey;
}

const _presets = [
  _Preset('Local vLLM', 'http://localhost:8000/v1', false),
  _Preset('OpenAI', 'https://api.openai.com/v1', true),
  _Preset('OpenRouter', 'https://openrouter.ai/api/v1', true),
  _Preset('Anthropic (OpenAI-compat)', 'https://api.anthropic.com/v1', true),
  _Preset('Custom', '', false),
];

class TeacherScreen extends StatefulWidget {
  const TeacherScreen({super.key, required this.runner});
  final JobRunner runner;
  @override
  State<TeacherScreen> createState() => _TeacherScreenState();
}

class _TeacherScreenState extends State<TeacherScreen> {
  _Preset preset = _presets.first;
  final url = TextEditingController(text: _presets.first.url);
  final key = TextEditingController();
  final model = TextEditingController();
  final input = TextEditingController();
  final output = TextEditingController();
  final prompt = TextEditingController();
  final dpi = TextEditingController(text: '200');
  final maxTok = TextEditingController(text: '8192');
  final workers = TextEditingController(text: '2');
  bool disableThinking = true;
  bool markdownPost = true;
  int mode = 0; // 0 = model, 1 = agent
  List<Map<String, dynamic>> promptPresets = [];
  String? promptPreset;
  List<String> models = [];
  String? msg;
  bool fetching = false;
  bool testing = false;
  Map<String, dynamic>? testResult;

  JobRunner get r => widget.runner;

  @override
  void initState() {
    super.initState();
    loadPrompts();
    r.addListener(_retryPrompts);
  }

  @override
  void dispose() {
    r.removeListener(_retryPrompts);
    super.dispose();
  }

  void _retryPrompts() {
    if (promptPresets.isEmpty && r.error == null) loadPrompts();
  }

  Future<void> loadPrompts() async {
    try {
      final l = await r.prompts();
      if (!mounted) return;
      setState(() {
        promptPresets = l;
        if (promptPreset == null && l.isNotEmpty && prompt.text.isEmpty) {
          promptPreset = l.first['name'];
          prompt.text = l.first['text'];
        }
      });
    } catch (_) {}
  }

  Future<void> fetchModels() async {
    setState(() {
      fetching = true;
      msg = null;
    });
    try {
      final ids = await r.listModels(url.text.trim(), key.text.trim());
      setState(() {
        models = ids;
        if (ids.isNotEmpty && model.text.isEmpty) model.text = ids.first;
        msg = 'Found ${ids.length} model(s).';
      });
    } catch (e) {
      setState(() => msg = 'Could not list models: $e');
    } finally {
      setState(() => fetching = false);
    }
  }

  Future<void> testConn() async {
    setState(() {
      testing = true;
      testResult = null;
    });
    Map<String, dynamic> res;
    try {
      if (url.text.trim().isEmpty || model.text.trim().isEmpty)
        throw 'Fill in Base URL and Model name first.';
      res = await r.testConnection(
        url.text.trim(),
        key.text.trim(),
        model.text.trim(),
      );
    } catch (e) {
      res = {'ok': false, 'error': '$e'};
    }
    if (mounted) {
      setState(() {
        testing = false;
        testResult = res;
      });
    }
  }

  Future<void> run() async {
    if (prompt.text.trim().isEmpty) {
      setState(() => msg = 'Write or pick a prompt first.');
      return;
    }
    if ([url, model, input, output].any((c) => c.text.trim().isEmpty)) {
      setState(
        () => msg = 'Fill in URL, model, input folder and output folder.',
      );
      return;
    }
    final script = '${r.root}/ovis_ft_ui/scripts/teacher_run.py';
    final cmd =
        'python3 ${shq(script)} ${shq(input.text.trim())} --out ${shq(output.text.trim())} '
        '--base-url ${shq(url.text.trim())} --model ${shq(model.text.trim())} '
        '--dpi ${int.tryParse(dpi.text) ?? 200} --max-tokens ${int.tryParse(maxTok.text) ?? 8192} --workers ${int.tryParse(workers.text) ?? 2}${disableThinking ? '' : ' --allow-thinking'} --postprocess ${markdownPost ? 'markdown' : 'none'}';
    final job = await r.start(
      'teacher',
      cmd,
      env: {
        'TEACHER_API_KEY': key.text.trim(),
        'TEACHER_PROMPT': prompt.text.trim(),
      },
    );
    setState(() => msg = 'Started in background (job ${job.id}).');
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: r,
      builder: (context, _) {
        final job = r.byName('teacher');
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
                    onSelectionChanged: (v) => setState(() => mode = v.first),
                  ),
                  gap(),
                  if (mode == 1) AgentForm(runner: r),
                  if (mode == 0) ...[
                    Section(
                      '1. Teacher model',
                      children: [
                        DropdownButtonFormField<_Preset>(
                          initialValue: preset,
                          decoration: const InputDecoration(
                            labelText: 'Provider',
                            border: OutlineInputBorder(),
                          ),
                          items: [
                            for (final p in _presets)
                              DropdownMenuItem(value: p, child: Text(p.name)),
                          ],
                          onChanged: (p) => setState(() {
                            preset = p!;
                            if (p.url.isNotEmpty) url.text = p.url;
                            models = [];
                          }),
                        ),
                        gap(),
                        TextField(
                          controller: url,
                          decoration: const InputDecoration(
                            labelText:
                                'Base URL (OpenAI-compatible, ends in /v1)',
                            border: OutlineInputBorder(),
                          ),
                        ),
                        gap(),
                        TextField(
                          controller: key,
                          obscureText: true,
                          decoration: InputDecoration(
                            labelText:
                                'API key${preset.needsKey ? '' : ' (optional for local vLLM)'}',
                            helperText: 'Passed to the job via environment only; never saved.',
                            border: const OutlineInputBorder(),
                          ),
                        ),
                        gap(),
                        Row(
                          children: [
                            Expanded(
                              child: TextField(
                                controller: model,
                                decoration: InputDecoration(
                                  labelText: 'Model name',
                                  border: const OutlineInputBorder(),
                                  suffixIcon: models.isEmpty
                                      ? null
                                      : PopupMenuButton<String>(
                                          icon: const Icon(
                                            Icons.arrow_drop_down,
                                          ),
                                          onSelected: (m) =>
                                              setState(() => model.text = m),
                                          itemBuilder: (_) => [
                                            for (final m in models)
                                              PopupMenuItem(
                                                value: m,
                                                child: Text(m),
                                              ),
                                          ],
                                        ),
                                ),
                              ),
                            ),
                            const SizedBox(width: 8),
                            FilledButton.tonalIcon(
                              onPressed: fetching ? null : fetchModels,
                              icon: const Icon(Icons.cloud_download_outlined),
                              label: const Text('Fetch models'),
                            ),
                          ],
                        ),
                        gap(),
                        Row(
                          children: [
                            FilledButton.icon(
                              onPressed: testing ? null : testConn,
                              icon: testing
                                  ? const SizedBox(
                                      width: 16,
                                      height: 16,
                                      child: CircularProgressIndicator(
                                        strokeWidth: 2,
                                      ),
                                    )
                                  : const Icon(Icons.wifi_tethering),
                              label: const Text(
                                'Test connection (hello world)',
                              ),
                            ),
                            gap(),
                            if (testResult != null)
                              Expanded(
                                child: Text(
                                  testResult!['ok'] == true
                                      ? '✓ Connected in ${testResult!['ms']} ms. Model replied: "${('${testResult!['reply'] ?? ''}').trim()}"'
                                      : '✗ ${testResult!['error']}',
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
                    Section(
                      '2. Prompt',
                      children: [
                        DropdownButtonFormField<String>(
                          key: ValueKey(promptPresets.length),
                          initialValue: promptPreset,
                          decoration: const InputDecoration(
                            labelText: 'Preset (editable below; add your own in scripts/prompts/*.txt)',
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
                    ),
                    Section(
                      '3. Input & output',
                      children: [
                        FolderField(
                          runner: r,
                          label: 'Input folder (PDFs or images, searched recursively)',
                          controller: input,
                        ),
                        gap(),
                        FolderField(
                          runner: r,
                          label: 'Output folder (one label file per input)',
                          controller: output,
                        ),
                        ExpansionTile(
                          tilePadding: EdgeInsets.zero,
                          title: const Text('Advanced'),
                          childrenPadding: const EdgeInsets.only(bottom: 8),
                          children: [
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
                                gap(),
                                Expanded(
                                  child: TextField(
                                    controller: workers,
                                    decoration: const InputDecoration(
                                      labelText: 'Parallel workers',
                                      border: OutlineInputBorder(),
                                    ),
                                  ),
                                ),
                              ],
                            ),
                            CheckboxListTile(
                              dense: true,
                              contentPadding: EdgeInsets.zero,
                              value: disableThinking,
                              onChanged: (v) =>
                                  setState(() => disableThinking = v ?? true),
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
                          ],
                        ),
                        gap(),
                        Row(
                          children: [
                            FilledButton.icon(
                              onPressed: r.isRunning('teacher') ? null : run,
                              icon: const Icon(Icons.play_arrow),
                              label: const Text('Run teacher in background'),
                            ),
                            gap(),
                            if (msg != null) Expanded(child: Text(msg!)),
                          ],
                        ),
                        const SizedBox(height: 4),
                        const Text(
                          'Already-finished files in the output folder are skipped, so re-running resumes.',
                          style: TextStyle(fontSize: 12),
                        ),
                      ],
                    ),
                  ],
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
