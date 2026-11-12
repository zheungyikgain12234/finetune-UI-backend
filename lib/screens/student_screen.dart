import 'package:flutter/material.dart';

import '../services/job_runner.dart';
import '../widgets/common.dart';

const _studentPresets = ['ATH-MaaS/OvisOCR2'];

class StudentScreen extends StatefulWidget {
  const StudentScreen({super.key, required this.runner});
  final JobRunner runner;
  @override
  State<StudentScreen> createState() => _StudentScreenState();
}

class _StudentScreenState extends State<StudentScreen> {
  final student = TextEditingController(text: _studentPresets.first);
  final pdfDir = TextEditingController();
  final teacherDir = TextEditingController();
  final foldsDir = TextEditingController();
  final k = TextEditingController(text: '3');
  final n = TextEditingController(text: '50');
  final seed = TextEditingController(text: '3407');
  final epochs = TextEditingController(text: '30');
  final lr = TextEditingController(text: '1e-4');
  final rank = TextEditingController(text: '32');
  final maxNew = TextEditingController(text: '8192');
  String? msg;

  JobRunner get r => widget.runner;

  List<Fold> foldList = [];

  Future<void> rescan() async {
    final l = await r.folds(foldsDir.text);
    if (mounted) setState(() => foldList = l);
  }

  @override
  void initState() {
    super.initState();
    r.addListener(rescan);
  }

  @override
  void dispose() {
    r.removeListener(rescan);
    super.dispose();
  }

  Future<void> makeFolds() async {
    if ([pdfDir, teacherDir, foldsDir].any((c) => c.text.trim().isEmpty)) {
      setState(
        () =>
            msg = 'Set the PDF folder, teacher output folder and folds folder.',
      );
      return;
    }
    final cmd =
        'python3 ${shq('${r.root}/ovis_ft_ui/scripts/make_folds_cli.py')} ${shq(pdfDir.text.trim())} '
        '${shq(teacherDir.text.trim())} --out ${shq(foldsDir.text.trim())} --k ${int.tryParse(k.text) ?? 3} '
        '--n ${int.tryParse(n.text) ?? 50} --seed ${int.tryParse(seed.text) ?? 3407} '
        '--build-dataset ${shq('${r.root}/build_dataset.py')}';
    await r.start('make-folds', cmd);
    setState(() => msg = 'Building folds in background…');
  }

  /// Train + infer for one fold (same steps as run_cross.sh).
  String foldCmd(Fold d) {
    final p = d.path;
    return 'set -e; '
        '[ -d ${shq('$p/outputs/merged_16bit')} ] || python3 train_ovis.py --data ${shq('$p/data')} '
        '--model ${shq(student.text.trim())} --epochs ${epochs.text.trim()} --lr ${lr.text.trim()} '
        '--rank ${rank.text.trim()} --out ${shq('$p/outputs')}; '
        'python3 ocr_folder.py ${shq('$p/test_pdfs')} --out ${shq('$p/predictions')} '
        '--model ${shq('$p/outputs/merged_16bit')} --data ${shq('$p/data')} --max-new ${maxNew.text.trim()}';
  }

  String foldName(Fold d) => 'train:${d.name}:${foldsDir.text.trim()}';

  Future<void> startFold(Fold d) async {
    if (r.jobs.any((j) => j.name.startsWith('train:') && j.running)) {
      setState(
        () => msg = 'A training job is already running; the GPU is shared, so wait or stop it first.',
      );
      return;
    }
    await r.start(foldName(d), foldCmd(d));
  }

  Future<void> startAll(List<Fold> fs) async {
    if (r.jobs.any((j) => j.name.startsWith('train:') && j.running)) {
      setState(() => msg = 'A training job is already running.');
      return;
    }
    final cmd = fs.map((d) => '( ${foldCmd(d)} )').join(' && ');
    await r.start('train:all:${foldsDir.text.trim()}', cmd);
  }

  String foldStatus(Fold d) {
    final j = r.byName(foldName(d));
    if (j != null && j.running) return 'running';
    if (d.status == 'ready' && j != null) return j.status;
    return d.status;
  }

  Widget f(String l, TextEditingController c) => SizedBox(
    width: 130,
    child: TextField(
      controller: c,
      decoration: InputDecoration(
        labelText: l,
        border: const OutlineInputBorder(),
      ),
    ),
  );

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
            Section(
              '1. Student model & hyper-parameters',
              children: [
                Autocomplete<String>(
                  initialValue: TextEditingValue(text: student.text),
                  optionsBuilder: (v) =>
                      _studentPresets.where((o) => o.contains(v.text)),
                  onSelected: (s) => student.text = s,
                  fieldViewBuilder: (ctx, c, fn, _) {
                    c.addListener(() => student.text = c.text);
                    return TextField(
                      controller: c,
                      focusNode: fn,
                      decoration: const InputDecoration(
                        labelText: 'Student model (HF id or local path)',
                        border: OutlineInputBorder(),
                      ),
                    );
                  },
                ),
                gap(),
                Wrap(
                  spacing: 12,
                  runSpacing: 12,
                  children: [
                    f('Epochs', epochs),
                    f('Learning rate', lr),
                    f('LoRA rank', rank),
                    f('Max new tokens', maxNew),
                  ],
                ),
              ],
            ),
            Section(
              '2. Cross-validation folds',
              children: [
                FolderField(
                  runner: r,
                  label: 'Source PDFs folder',
                  controller: pdfDir,
                ),
                gap(),
                FolderField(
                  runner: r,
                  label: 'Teacher output folder (.md)',
                  controller: teacherDir,
                ),
                gap(),
                FolderField(
                  runner: r,
                  label: 'Folds folder (will contain fold_1, fold_2, …)',
                  controller: foldsDir,
                ),
                gap(),
                Wrap(
                  spacing: 12,
                  runSpacing: 12,
                  crossAxisAlignment: WrapCrossAlignment.center,
                  children: [
                    f('Folds (k)', k),
                    f('Test pages / fold', n),
                    f('Seed', seed),
                    FilledButton.icon(
                      onPressed: r.isRunning('make-folds') ? null : makeFolds,
                      icon: const Icon(Icons.call_split),
                      label: const Text('Create folds'),
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
              '3. Fine-tune folds',
              children: [
                Row(
                  children: [
                    FilledButton.icon(
                      onPressed: fs.isEmpty ? null : () => startAll(fs),
                      icon: const Icon(Icons.playlist_play),
                      label: const Text('Start all folds (sequential)'),
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
                    'No folds found yet. Create them above, or point "Folds folder" at existing ones.',
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
                  'Training skips a fold whose merged model already exists, so interrupted runs resume at the inference step.',
                  style: TextStyle(fontSize: 12),
                ),
              ],
            ),
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
