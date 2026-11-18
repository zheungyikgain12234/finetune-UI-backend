import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';

import '../services/job_runner.dart';
import '../widgets/common.dart';

class StudentScreen extends StatefulWidget {
  const StudentScreen({super.key, required this.runner});
  final JobRunner runner;
  @override
  State<StudentScreen> createState() => _StudentScreenState();
}

class _StudentScreenState extends State<StudentScreen> {
  final student = TextEditingController();
  final sourceDir = TextEditingController();
  final teacherDir = TextEditingController();
  final foldsDir = TextEditingController();
  final k = TextEditingController(text: '3');
  final n = TextEditingController(text: '50');
  final seed = TextEditingController(text: '3407');
  final params = <String, TextEditingController>{};
  final buildCmd = TextEditingController();
  final trainCmd = TextEditingController();
  final inferCmd = TextEditingController();
  final doneMarker = TextEditingController(text: 'outputs');
  List<Map<String, dynamic>> recipes = [];
  String? recipeName;
  String? msg;

  // Quick mode: pick a model, upload zips, train. Everything lives in a server-side workspace (no folder paths).
  bool quick = true;
  List<Map<String, dynamic>> models = [];
  String? qModelId;
  final qPrompt = TextEditingController(
    text: 'Convert the input into the JSON described by the training examples. Output only the JSON.',
  );
  final qHoldout = TextEditingController(text: '10');
  final qLr = TextEditingController(text: '1e-4');
  final qRank = TextEditingController(text: '32');
  final qMaxLen = TextEditingController(text: '8192');
  final qMaxNew = TextEditingController(text: '2048');
  bool sepTest = false;
  double qEpochs = 3;
  Map<String, dynamic> ws = {};
  String? uploading;
  final uploadMsg = <String, String>{};

  Future<void> loadModels() async {
    try {
      final l = await r.allowedModels();
      if (!mounted) return;
      setState(() {
        models = l;
        qModelId ??= l.isNotEmpty ? l.first['id'] as String : null;
      });
    } catch (_) {}
  }

  Future<void> refreshWs() async {
    if (r.wid.isEmpty || r.error != null) return;
    try {
      final w = await r.wsStatus();
      if (mounted) setState(() => ws = w);
    } catch (_) {}
  }

  Future<void> pickZip(String slot) async {
    try {
      final f = await FilePicker.pickFile(
        type: FileType.custom,
        allowedExtensions: ['zip'],
        dialogTitle: 'Choose a .zip',
      );
      if (f == null) return;
      setState(() {
        uploading = slot;
        uploadMsg.remove(slot);
      });
      final res = await r.uploadZip(slot, await f.readAsBytes());
      uploadMsg[slot] =
          '${f.name}: ${res['files']} file(s)${res['skipped'] > 0 ? ', ${res['skipped']} skipped (unsupported or unsafe)' : ''}';
      await refreshWs();
    } catch (e) {
      uploadMsg[slot] = 'Upload failed: $e';
    } finally {
      if (mounted) setState(() => uploading = null);
    }
  }

  /// One upload tile: what to put in the zip, a button, and what the server received.
  Widget _uploadTile(String slot, String title, String hint) {
    final info = (ws['slots'] as Map?)?[slot] as Map?;
    final types = info == null ? '' : (info['types'] as Map).entries.map((e) => '${e.value}× ${e.key}').join(', ');
    return ListTile(
      dense: true,
      contentPadding: EdgeInsets.zero,
      leading: Icon(info != null ? Icons.check_circle : Icons.upload_file, color: info != null ? Colors.green : null),
      title: Text(title),
      subtitle: Text(uploadMsg[slot] ?? (info != null ? 'Uploaded: ${info['files']} file(s) ($types)' : hint)),
      trailing: OutlinedButton.icon(
        onPressed: uploading != null ? null : () => pickZip(slot),
        icon: uploading == slot
            ? const SizedBox(width: 14, height: 14, child: CircularProgressIndicator(strokeWidth: 2))
            : const Icon(Icons.folder_zip_outlined),
        label: Text(info != null ? 'Replace .zip' : 'Choose .zip'),
      ),
    );
  }

  JobRunner get r => widget.runner;

  List<Fold> foldList = [];

  Future<void> loadRecipes() async {
    try {
      final l = await r.recipes();
      if (!mounted) return;
      setState(() {
        recipes = l;
        if (recipeName == null && l.isNotEmpty) applyRecipe(l.first);
      });
    } catch (_) {}
  }

  void applyRecipe(Map<String, dynamic> rc) {
    recipeName = rc['name'];
    student.text = rc['student'] ?? '';
    buildCmd.text = rc['buildCmd'] ?? '';
    trainCmd.text = rc['trainCmd'] ?? '';
    inferCmd.text = rc['inferCmd'] ?? '';
    doneMarker.text = rc['doneMarker'] ?? 'outputs';
    params.clear();
    (rc['params'] as Map? ?? {}).forEach(
      (k, v) => params[k] = TextEditingController(text: '$v'),
    );
  }

  Future<void> addParam() async {
    final c = TextEditingController();
    final name = await showDialog<String>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('New hyper-parameter'),
        content: TextField(
          controller: c,
          autofocus: true,
          decoration: const InputDecoration(
            labelText: 'Name, used as {name} in the commands',
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(ctx, c.text.trim()),
            child: const Text('Add'),
          ),
        ],
      ),
    );
    if (name != null && RegExp(r'^[A-Za-z_]\w*$').hasMatch(name)) {
      setState(() => params.putIfAbsent(name, () => TextEditingController()));
    }
  }

  Future<void> rescan() async {
    if (quick) refreshWs();
    if (recipes.isEmpty && r.error == null) loadRecipes();
    final l = await r.folds(
      foldsDir.text,
      marker: doneMarker.text.trim().isEmpty
          ? 'outputs'
          : doneMarker.text.trim(),
    );
    if (mounted) setState(() => foldList = l);
  }

  @override
  void initState() {
    super.initState();
    loadModels();
    loadRecipes();
    r.addListener(rescan);
  }

  @override
  void dispose() {
    r.removeListener(rescan);
    super.dispose();
  }

  /// Replaces {name} with the shell-quoted value; reports placeholders it does not know.
  String fill(String tpl, Map<String, String> vals) {
    final all = {
      ...{for (final e in params.entries) e.key: e.value.text.trim()},
      ...vals,
    };
    return tpl.replaceAllMapped(RegExp(r'\{(\w+)\}'), (m) {
      final v = all[m[1]];
      if (v == null)
        throw 'Unknown placeholder {${m[1]}} in a command template.';
      return shq(v);
    });
  }

  Future<void> makeFolds() async {
    if ([sourceDir, teacherDir, foldsDir].any((c) => c.text.trim().isEmpty)) {
      setState(
        () => msg =
            'Set the source folder, teacher output folder and folds folder.',
      );
      return;
    }
    final cmd =
        'python3 ${shq('${r.root}/ovis_ft_ui/scripts/make_folds_cli.py')} ${shq(sourceDir.text.trim())} '
        '${shq(teacherDir.text.trim())} --out ${shq(foldsDir.text.trim())} --k ${int.tryParse(k.text) ?? 3} '
        '--n ${int.tryParse(n.text) ?? 50} --seed ${int.tryParse(seed.text) ?? 3407} '
        '--build-cmd ${shq(buildCmd.text.trim())}';
    await r.start('make-folds', cmd);
    setState(() => msg = 'Building folds in background…');
  }

  /// Train (skipped if the done-marker exists) then infer, for one fold. Built from the recipe templates.
  String foldCmd(Fold d) {
    final p = d.path;
    final vals = {
      'fold': p,
      'data': '$p/data',
      'outputs': '$p/outputs',
      'predictions': '$p/predictions',
      'test_inputs': '$p/test_inputs',
      'model': student.text.trim(),
    };
    final marker = doneMarker.text.trim().isEmpty
        ? 'outputs'
        : doneMarker.text.trim();
    final train = fill(trainCmd.text.trim(), vals);
    final infer = inferCmd.text.trim().isEmpty
        ? ''
        : '; ${fill(inferCmd.text.trim(), vals)}';
    return 'set -e; [ -e ${shq('$p/$marker')} ] || { $train; }$infer';
  }

  String foldName(Fold d) => 'train:${d.name}:${foldsDir.text.trim()}';

  Future<void> startFold(Fold d) async {
    if (r.jobs.any((j) => j.name.startsWith('train:') && j.running)) {
      setState(
        () => msg = 'A training job is already running; the GPU is shared, so wait or stop it first.',
      );
      return;
    }
    try {
      await r.start(foldName(d), foldCmd(d));
    } catch (e) {
      setState(() => msg = '$e');
    }
  }

  Future<void> startAll(List<Fold> fs) async {
    if (r.jobs.any((j) => j.name.startsWith('train:') && j.running)) {
      setState(() => msg = 'A training job is already running.');
      return;
    }
    try {
      final cmd = fs.map((d) => '( ${foldCmd(d)} )').join(' && ');
      await r.start('train:all:${foldsDir.text.trim()}', cmd);
    } catch (e) {
      setState(() => msg = '$e');
    }
  }

  String foldStatus(Fold d) {
    final j = r.byName(foldName(d));
    if (j != null && j.running) return 'running';
    if (d.status == 'ready' && j != null) return j.status;
    return d.status;
  }

  Widget _cmdField(String label, TextEditingController c) => TextField(
    controller: c,
    minLines: 2,
    maxLines: 6,
    style: const TextStyle(fontFamily: 'monospace', fontSize: 12),
    decoration: InputDecoration(
      labelText: label,
      alignLabelWithHint: true,
      border: const OutlineInputBorder(),
    ),
  );

  static const _hints = {
    'epochs': 'How many passes over the training data. More = learns longer (and slower).',
    'lr': 'Learning rate: how big each update is. Leave the default unless results are unstable.',
    'rank': 'LoRA rank: capacity of the adapter. Higher = more capacity, more GPU memory.',
    'max_new': 'Longest output (in tokens) the model may produce per page.',
  };

  Widget f(String l, TextEditingController c, {String? hint}) => SizedBox(
    width: 150,
    child: Tooltip(
      message: hint ?? _hints[l] ?? '',
      child: TextField(
        controller: c,
        decoration: InputDecoration(
          labelText: l,
          border: const OutlineInputBorder(),
        ),
      ),
    ),
  );

  Widget help(String t) => Padding(
    padding: const EdgeInsets.only(bottom: 4),
    child: Text(t, style: Theme.of(context).textTheme.bodySmall),
  );

  Widget get intro => Card(
    color: Theme.of(context).colorScheme.secondaryContainer,
    margin: const EdgeInsets.only(bottom: 16),
    child: const Padding(
      padding: EdgeInsets.all(16),
      child: Text(
        'What this tab does: it teaches a smaller "student" model to copy the teacher\'s labels.\n'
        '  1. Pick a recipe (the defaults are fine to start).\n'
        '  2. Split your files into folds: each fold trains on most files and is tested on a few it never saw.\n'
        '  3. Press Start to fine-tune on each fold and check the result.\n'
        'Tip: run Teacher first so there are labels to learn from. Only one training job runs at a time.',
      ),
    ),
  );

  String get qJob => 'ws:${r.wid}:all';
  String qTestJob(String step) => 'ws:${r.wid}:$step';
  bool get _busy => r.jobs.any((x) => x.name.startsWith('ws:') && x.running);

  /// One row per pipeline step: what it checks, and a Test button that runs just that step as a small job.
  Widget _testRow(String step, String title, String what) {
    final j = r.byName(qTestJob(step));
    final icon = j == null
        ? Icons.radio_button_unchecked
        : j.running
        ? Icons.hourglass_top
        : j.status == 'done'
        ? Icons.check_circle
        : Icons.error;
    return ListTile(
      dense: true,
      contentPadding: EdgeInsets.zero,
      leading: Icon(icon),
      title: Text(title),
      subtitle: Text(j == null ? what : '$what\nLast test: ${j.status}'),
      isThreeLine: j != null,
      trailing: Wrap(
        spacing: 4,
        children: [
          OutlinedButton(
            onPressed: _busy ? null : () => startQuick(step),
            child: const Text('Test'),
          ),
          if (j != null)
            TextButton(
              onPressed: () => showLogDialog(context, r, j),
              child: const Text('Logs'),
            ),
        ],
      ),
    );
  }

  Future<void> startQuick([String step = 'all']) async {
    if (qModelId == null || qPrompt.text.trim().isEmpty) {
      setState(() => msg = 'Pick a model and write an instruction.');
      return;
    }
    if ((ws['slots'] as Map?)?['inputs'] == null) {
      setState(() => msg = 'Upload the inputs .zip first.');
      return;
    }
    if (sepTest && (ws['slots'] as Map?)?['test_inputs'] == null) {
      setState(() => msg = 'Upload the test inputs .zip, or switch to "Hold out part of the data".');
      return;
    }
    try {
      await r.wsRun({
        'step': step,
        'model': qModelId,
        'prompt': qPrompt.text.trim(),
        'holdout': double.tryParse(qHoldout.text) ?? 10,
        'epochs': qEpochs.round(),
        'lr': double.tryParse(qLr.text) ?? 1e-4,
        'rank': int.tryParse(qRank.text) ?? 32,
        'max_len': int.tryParse(qMaxLen.text) ?? 8192,
        'max_new': int.tryParse(qMaxNew.text) ?? 2048,
      });
      setState(() => msg = 'Started. Open Logs to watch progress.');
    } catch (e) {
      setState(() => msg = '$e');
    }
  }

  Widget get modeToggle => Padding(
    padding: const EdgeInsets.only(bottom: 16),
    child: SegmentedButton<bool>(
      segments: const [
        ButtonSegment(
          value: true,
          icon: Icon(Icons.bolt),
          label: Text('Quick start'),
        ),
        ButtonSegment(
          value: false,
          icon: Icon(Icons.tune),
          label: Text('Advanced (recipes & folds)'),
        ),
      ],
      selected: {quick},
      onSelectionChanged: (v) => setState(() {
        quick = v.first;
        msg = null;
      }),
    ),
  );

  Widget _downloads() {
    if (ws['lora'] != true) return const SizedBox.shrink();
    final score = ws['summary'] is Map ? (ws['summary'] as Map)['mean_score'] : null;
    return Card(
      margin: const EdgeInsets.only(top: 16),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Your fine-tuned model', style: Theme.of(context).textTheme.titleSmall),
            if (score != null) Text('Score on the held-out test files: ${(score as num).toStringAsFixed(3)}'),
            gap(8),
            const Text(
              'Download the LoRA adapter (small). It includes a README that shows how to use it with the base model, '
              'or merge it into one model on your own machine, plus the instruction you trained with.',
            ),
            gap(8),
            Wrap(
              spacing: 12,
              runSpacing: 8,
              children: [
                FilledButton.icon(
                  onPressed: () => r.download('lora'),
                  icon: const Icon(Icons.download),
                  label: Text('Download LoRA adapter (${ws['loraMB']} MB)'),
                ),
                if (ws['merged'] == true)
                  OutlinedButton.icon(
                    onPressed: () => r.download('merged'),
                    icon: const Icon(Icons.archive_outlined),
                    label: Text('Merged model (${ws['mergedGB']} GB)'),
                  ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  List<Widget> _quickSections() {
    final j = r.byName(qJob);
    final vision = models.where((m) => m['id'] == qModelId).any((m) => m['vision'] == true);
    return [
      Card(
        color: Theme.of(context).colorScheme.secondaryContainer,
        margin: const EdgeInsets.only(bottom: 16),
        child: const Padding(
          padding: EdgeInsets.all(16),
          child: Text(
            'Teach a model to turn INPUT files into LABEL files: Markdown → JSON, image → text, PDF → Markdown, text → text.\n'
            '1. Pick a model and write an instruction.  2. Upload a .zip of inputs and a .zip of labels '
            '(files are paired by name: a.md ↔ a.json; one zip with both also works).  3. Choose how to hold out test files.  '
            '4. Test each step, then Start.\n'
            'After training you can download the LoRA adapter.',
          ),
        ),
      ),
      Section(
        'Fine-tune a student model',
        children: [
          DropdownButtonFormField<String>(
            initialValue: qModelId,
            isExpanded: true,
            items: [
              for (final m in models)
                DropdownMenuItem(value: m['id'] as String, child: Text('${m['label'] ?? m['id']}')),
            ],
            onChanged: (v) => setState(() => qModelId = v),
            decoration: InputDecoration(
              labelText: 'Model to fine-tune',
              helperText: models.isEmpty
                  ? 'Loading the model list from the server…'
                  : qModelId ?? '',
              border: const OutlineInputBorder(),
            ),
          ),
          gap(),
          TextField(
            controller: qPrompt,
            minLines: 2,
            maxLines: 6,
            decoration: const InputDecoration(
              labelText: 'Instruction: what should the model do with each input?',
              helperText: 'Shown to the model before every input, in training and at test time. Be specific about the output format.',
              border: OutlineInputBorder(),
            ),
          ),
          gap(),
          _uploadTile(
            'inputs',
            'Inputs (.zip)',
            'What the model reads: .md .txt .json${vision ? ', images or PDFs' : ''}.',
          ),
          _uploadTile(
            'labels',
            'Labels (.zip, optional if they are in the inputs zip)',
            'What it should produce: .json .md .txt, same file names as the inputs.',
          ),
          gap(),
          SegmentedButton<bool>(
            segments: const [
              ButtonSegment(value: false, label: Text('Hold out part of the data as test')),
              ButtonSegment(value: true, label: Text('Upload separate test files')),
            ],
            selected: {sepTest},
            onSelectionChanged: (v) => setState(() => sepTest = v.first),
          ),
          gap(),
          if (!sepTest)
            SizedBox(
              width: 260,
              child: TextField(
                controller: qHoldout,
                decoration: const InputDecoration(
                  labelText: 'Test share (%)',
                  helperText: 'Random files kept out of training and used to score the model.',
                  border: OutlineInputBorder(),
                ),
              ),
            )
          else ...[
            _uploadTile('test_inputs', 'Test inputs (.zip, never trained on)', 'Same kind of files as the inputs.'),
            _uploadTile('test_labels', 'Test labels (.zip, optional)', 'Leave empty if they are in the test inputs zip.'),
          ],
          gap(),
          Text(
            'Test each step first (quick, nothing is trained)',
            style: Theme.of(context).textTheme.titleSmall,
          ),
          _testRow('check', '1. Files', 'Pairs inputs with labels by file name, shows the file types, checks train and test do not overlap.'),
          _testRow('build', '2. Dataset', 'Builds the training dataset (train / validation / test files) from your pairs.'),
          _testRow('check-train', '3. Model & training format', 'Loads the model and tokenizes one example; the trained-on text must be your label, not your input.'),
          _testRow('check-infer', '4. Inference', 'Runs the original (not yet fine-tuned) model on 1 test example and prints its answer.'),
          const Divider(),
          Text('Epochs: ${qEpochs.round()}', style: Theme.of(context).textTheme.titleSmall),
          help('One epoch = one full pass over the train set. More epochs learn longer but take longer; 3 is a safe start.'),
          Slider(
            value: qEpochs,
            min: 1,
            max: 30,
            divisions: 29,
            label: '${qEpochs.round()}',
            onChanged: (v) => setState(() => qEpochs = v),
          ),
          gap(),
          Wrap(
            spacing: 12,
            runSpacing: 12,
            crossAxisAlignment: WrapCrossAlignment.center,
            children: [
              FilledButton.icon(
                onPressed: _busy ? null : () => startQuick(),
                icon: const Icon(Icons.play_arrow),
                label: const Text('Start fine-tuning'),
              ),
              if (j != null)
                OutlinedButton.icon(
                  onPressed: () => showLogDialog(context, r, j),
                  icon: const Icon(Icons.terminal),
                  label: Text('Logs: ${j.status}'),
                ),
              if (j != null && j.running)
                OutlinedButton.icon(
                  onPressed: () => r.stop(j),
                  icon: const Icon(Icons.stop_circle_outlined, color: Colors.redAccent),
                  label: const Text('Stop'),
                ),
            ],
          ),
          if (msg != null) Padding(padding: const EdgeInsets.only(top: 8), child: Text(msg!)),
          _downloads(),
        ],
      ),
    ];
  }

  bool get canMake =>
      [sourceDir, teacherDir, foldsDir].every((c) => c.text.trim().isNotEmpty);

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: r,
      builder: (context, _) {
        final fs = foldList;
        final all = r.byName('train:all:${foldsDir.text.trim()}');
        final mk = r.byName('make-folds');
        return ListView(
          padding: const EdgeInsets.all(16),
          children: [
            // modeToggle hidden for now: the Advanced (recipes & folds) tab is ignored for students.
            // To bring it back, restore `modeToggle,` here.
            if (quick)
              ..._quickSections()
            else ...[
              intro,
              Section(
                '1. Choose a recipe & settings',
                children: [
                  help(
                    'A recipe is a ready-made setup: which model to train and how. Picking one fills in everything below.',
                  ),
                  DropdownButtonFormField<String>(
                    key: ValueKey(recipes.length),
                    initialValue: recipeName,
                    decoration: const InputDecoration(
                      labelText: 'Recipe',
                      border: OutlineInputBorder(),
                    ),
                    items: [
                      for (final rc in recipes)
                        DropdownMenuItem(
                          value: rc['name'] as String,
                          child: Text(rc['name']),
                        ),
                    ],
                    onChanged: (n) => setState(
                      () => applyRecipe(
                        recipes.firstWhere((x) => x['name'] == n),
                      ),
                    ),
                  ),
                  if (recipes.any(
                    (x) => x['name'] == recipeName && x['description'] != null,
                  ))
                    Padding(
                      padding: const EdgeInsets.only(top: 6),
                      child: Text(
                        recipes.firstWhere(
                          (x) => x['name'] == recipeName,
                        )['description'],
                        style: Theme.of(context).textTheme.bodySmall,
                      ),
                    ),
                  gap(),
                  TextField(
                    controller: student,
                    decoration: const InputDecoration(
                      labelText: 'Student model',
                      helperText: 'The model to fine-tune: a Hugging Face id or a local folder path.',
                      border: OutlineInputBorder(),
                    ),
                  ),
                  gap(),
                  help(
                    'Training settings (hover a box for what it means). The defaults usually work.',
                  ),
                  Wrap(
                    spacing: 12,
                    runSpacing: 12,
                    crossAxisAlignment: WrapCrossAlignment.center,
                    children: [
                      for (final e in params.entries) f(e.key, e.value),
                      OutlinedButton.icon(
                        onPressed: addParam,
                        icon: const Icon(Icons.add),
                        label: const Text('Add parameter'),
                      ),
                    ],
                  ),
                  ExpansionTile(
                    tilePadding: EdgeInsets.zero,
                    title: const Text(
                      'Advanced: commands (most people can skip this)',
                    ),
                    childrenPadding: const EdgeInsets.only(bottom: 8),
                    children: [
                      const Align(
                        alignment: Alignment.centerLeft,
                        child: Text(
                          'Run from the workspace folder with the venv active. Placeholders: {model} {fold} {data} {outputs} '
                          '{predictions} {test_inputs}, your hyper-parameters, and for the dataset command {source_dir} '
                          '{labels_dir} {out} {test_list}. Values are shell-quoted for you.',
                          style: TextStyle(fontSize: 12),
                        ),
                      ),
                      gap(),
                      _cmdField(
                        'Dataset build command (run once per fold)',
                        buildCmd,
                      ),
                      gap(),
                      _cmdField('Training command', trainCmd),
                      gap(),
                      _cmdField(
                        'Inference command on held-out files (optional)',
                        inferCmd,
                      ),
                      gap(),
                      TextField(
                        controller: doneMarker,
                        decoration: const InputDecoration(
                          labelText: 'Done marker, relative to the fold folder (training is skipped if it exists)',
                          border: OutlineInputBorder(),
                        ),
                      ),
                    ],
                  ),
                ],
              ),
              Section(
                '2. Split data into folds',
                children: [
                  help(
                    'Each fold keeps some files hidden from training so you can measure how well the student does on unseen files. '
                    'Pick 3 folders, then press Create folds.',
                  ),
                  FolderField(
                    runner: r,
                    label: '① Original files (the PDFs / images you gave the teacher)',
                    controller: sourceDir,
                  ),
                  gap(),
                  FolderField(
                    runner: r,
                    label: '② Teacher results (the labels to learn from)',
                    controller: teacherDir,
                  ),
                  gap(),
                  FolderField(
                    runner: r,
                    label: '③ Where to put the folds (an empty folder; fold_1, fold_2, … are created inside)',
                    controller: foldsDir,
                  ),
                  gap(),
                  Wrap(
                    spacing: 12,
                    runSpacing: 12,
                    crossAxisAlignment: WrapCrossAlignment.center,
                    children: [
                      f(
                        'Folds (k)',
                        k,
                        hint: 'How many train/test splits to make. 3 is a good start.',
                      ),
                      f(
                        'Test files / fold',
                        n,
                        hint: 'Files hidden from training in each fold and used to check the result.',
                      ),
                      f(
                        'Seed',
                        seed,
                        hint: 'Random seed. Keep it the same to get the same split again.',
                      ),
                      FilledButton.icon(
                        onPressed: r.isRunning('make-folds') ? null : makeFolds,
                        icon: const Icon(Icons.call_split),
                        label: const Text('Create folds'),
                      ),
                      if (!canMake)
                        Text(
                          'Fill in all three folders first.',
                          style: Theme.of(context).textTheme.bodySmall,
                        ),
                      if (mk != null)
                        TextButton(
                          onPressed: () => showLogDialog(context, r, mk),
                          child: Text('make-folds: ${mk.status}'),
                        ),
                    ],
                  ),
                ],
              ),
              Section(
                '3. Train & test',
                children: [
                  help(
                    'Start trains the student on a fold, then runs it on that fold\'s hidden test files. Results appear in the fold\'s predictions folder.',
                  ),
                  Row(
                    children: [
                      FilledButton.icon(
                        onPressed: fs.isEmpty ? null : () => startAll(fs),
                        icon: const Icon(Icons.playlist_play),
                        label: const Text('Start all folds, one after another'),
                      ),
                      gap(),
                      if (all != null)
                        OutlinedButton.icon(
                          onPressed: () => showLogDialog(context, r, all),
                          icon: const Icon(Icons.terminal),
                          label: Text('Logs: ${all.status}'),
                        ),
                      gap(),
                      OutlinedButton.icon(
                        onPressed: rescan,
                        icon: const Icon(Icons.refresh),
                        label: const Text('Rescan folds'),
                      ),
                    ],
                  ),
                  gap(),
                  if (fs.isEmpty)
                    const Text(
                      'No folds yet. Finish step 2 (Create folds), or point box ③ at a folder that already has them.',
                    ),
                  for (final d in fs)
                    Builder(
                      builder: (_) {
                        final j = r.byName(foldName(d));
                        return ListTile(
                          dense: true,
                          leading: const Icon(Icons.dataset),
                          title: Text(d.name),
                          subtitle: Text('status: ${foldStatus(d)}'),
                          trailing: Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              TextButton.icon(
                                onPressed: () => startFold(d),
                                icon: const Icon(Icons.play_arrow),
                                label: const Text('Start'),
                              ),
                              TextButton.icon(
                                onPressed: j == null
                                    ? null
                                    : () => showLogDialog(context, r, j),
                                icon: const Icon(Icons.terminal),
                                label: const Text('Logs'),
                              ),
                              if (j != null && j.running)
                                IconButton(
                                  tooltip: 'Stop',
                                  icon: const Icon(
                                    Icons.stop_circle_outlined,
                                    color: Colors.redAccent,
                                  ),
                                  onPressed: () => r.stop(j),
                                ),
                            ],
                          ),
                        );
                      },
                    ),
                  if (msg != null)
                    Padding(
                      padding: const EdgeInsets.only(top: 8),
                      child: Text(msg!),
                    ),
                  const SizedBox(height: 4),
                  const Text(
                    'Safe to re-run: folds that already finished training are not trained again.',
                    style: TextStyle(fontSize: 12),
                  ),
                ],
              ),
            ],
            Section(
              'Jobs',
              children: [
                if (r.jobs.isEmpty) const Text('No jobs yet.'),
                for (final j in r.jobs.reversed.take(15))
                  ListTile(
                    dense: true,
                    title: Text(j.name, overflow: TextOverflow.ellipsis),
                    subtitle: Text(
                      '${j.startedAt.toString().split('.').first}  •  ${j.status}',
                    ),
                    trailing: TextButton(
                      onPressed: () => showLogDialog(context, r, j),
                      child: const Text('Tail logs'),
                    ),
                  ),
              ],
            ),
          ],
        );
      },
    );
  }
}
