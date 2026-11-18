import 'dart:async';

import 'package:flutter/material.dart';

import '../services/job_runner.dart';

/// Path text field with a server-side folder browser (the browser itself cannot expose real paths).
class FolderField extends StatelessWidget {
  const FolderField({
    super.key,
    required this.label,
    required this.controller,
    required this.runner,
    this.hint,
  });
  final String label;
  final String? hint;
  final TextEditingController controller;
  final JobRunner runner;

  @override
  Widget build(BuildContext context) => Row(
    children: [
      Expanded(
        child: TextField(
          controller: controller,
          decoration: InputDecoration(
            labelText: label,
            hintText: hint ?? '/home/…/folder',
            border: const OutlineInputBorder(),
          ),
        ),
      ),
      const SizedBox(width: 8),
      IconButton.filledTonal(
        tooltip: 'Browse folders',
        icon: const Icon(Icons.folder_open),
        onPressed: () async {
          final p = await showDialog<String>(
            context: context,
            builder: (_) =>
                FolderPickerDialog(runner: runner, start: controller.text),
          );
          if (p != null) controller.text = p;
        },
      ),
    ],
  );
}

class FolderPickerDialog extends StatefulWidget {
  const FolderPickerDialog({
    super.key,
    required this.runner,
    required this.start,
  });
  final JobRunner runner;
  final String start;
  @override
  State<FolderPickerDialog> createState() => _FolderPickerDialogState();
}

class _FolderPickerDialogState extends State<FolderPickerDialog> {
  Map<String, dynamic>? d;
  String? err;
  final pathCtl = TextEditingController();

  @override
  void initState() {
    super.initState();
    go(widget.start);
  }

  Future<void> go(String p) async {
    try {
      final r = await widget.runner.ls(p);
      setState(() {
        d = r;
        err = null;
        pathCtl.text = r['path'];
      });
    } catch (e) {
      setState(() => err = 'Backend error: $e');
    }
  }

  Future<void> newFolder() async {
    final c = TextEditingController();
    final name = await showDialog<String>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('New folder'),
        content: TextField(
          controller: c,
          autofocus: true,
          decoration: const InputDecoration(labelText: 'Name'),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(ctx, c.text.trim()),
            child: const Text('Create'),
          ),
        ],
      ),
    );
    if (name == null || name.isEmpty) return;
    try {
      final np = '${d!['path']}/$name';
      await widget.runner.mkdir(np);
      await go(np);
    } catch (e) {
      setState(() => err = '$e');
    }
  }

  @override
  Widget build(BuildContext context) {
    final dirs = (d?['dirs'] as List?)?.cast<String>() ?? [];
    return AlertDialog(
      title: const Text('Select folder'),
      content: SizedBox(
        width: 560,
        height: 440,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Row(
              children: [
                IconButton(
                  tooltip: 'Up',
                  icon: const Icon(Icons.arrow_upward),
                  onPressed: d == null ? null : () => go(d!['parent']),
                ),
                Expanded(
                  child: TextField(
                    controller: pathCtl,
                    onSubmitted: go,
                    decoration: const InputDecoration(
                      isDense: true,
                      border: OutlineInputBorder(),
                    ),
                  ),
                ),
              ],
            ),
            if (err != null)
              Text(err!, style: const TextStyle(color: Colors.redAccent)),
            if (d != null)
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 6),
                child: Text(
                  '${d!['pdfs']} PDF/image file(s), ${d!['mds']} .md file(s) in this folder',
                  style: Theme.of(context).textTheme.bodySmall,
                ),
              ),
            Expanded(
              child: ListView(
                children: [
                  for (final n in dirs)
                    ListTile(
                      dense: true,
                      leading: const Icon(Icons.folder),
                      title: Text(n),
                      onTap: () => go('${d!['path']}/$n'),
                    ),
                  if (d != null && dirs.isEmpty)
                    const ListTile(dense: true, title: Text('(no subfolders)')),
                ],
              ),
            ),
          ],
        ),
      ),
      actions: [
        TextButton.icon(
          onPressed: d == null ? null : newFolder,
          icon: const Icon(Icons.create_new_folder),
          label: const Text('New folder'),
        ),
        TextButton(
          onPressed: () => Navigator.pop(context),
          child: const Text('Cancel'),
        ),
        FilledButton(
          onPressed: d == null
              ? null
              : () => Navigator.pop(context, d!['path'] as String),
          child: const Text('Select this folder'),
        ),
      ],
    );
  }
}

Widget gap([double h = 12]) => SizedBox(height: h, width: h);

class Section extends StatelessWidget {
  const Section(this.title, {super.key, required this.children});
  final String title;
  final List<Widget> children;
  @override
  Widget build(BuildContext context) => Card(
    margin: const EdgeInsets.only(bottom: 16),
    child: Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(title, style: Theme.of(context).textTheme.titleMedium),
          gap(),
          ...children,
        ],
      ),
    ),
  );
}

/// Live log viewer for one job. "Live" polls every 2s; "Refresh" is a manual tail.
class LogViewer extends StatefulWidget {
  const LogViewer({super.key, required this.runner, required this.job});
  final JobRunner runner;
  final Job job;
  @override
  State<LogViewer> createState() => _LogViewerState();
}

class _LogViewerState extends State<LogViewer> {
  Timer? _t;
  bool live = true;
  String text = '';
  final scroll = ScrollController();

  @override
  void initState() {
    super.initState();
    _refresh();
    _setLive(true);
  }

  @override
  void didUpdateWidget(LogViewer old) {
    super.didUpdateWidget(old);
    if (old.job.id != widget.job.id) _refresh();
  }

  void _setLive(bool v) {
    live = v;
    _t?.cancel();
    if (v) _t = Timer.periodic(const Duration(seconds: 2), (_) => _refresh());
  }

  Future<void> _refresh() async {
    final t = await widget.runner.tail(widget.job);
    if (!mounted) return;
    setState(() => text = t);
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (scroll.hasClients) scroll.jumpTo(scroll.position.maxScrollExtent);
    });
  }

  @override
  void dispose() {
    _t?.cancel();
    scroll.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final j = widget.job;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Row(
          children: [
            Expanded(
              child: Text(
                '${j.name}  —  ${j.status}',
                style: Theme.of(context).textTheme.titleSmall,
              ),
            ),
            const Text('Live'),
            Switch(value: live, onChanged: (v) => setState(() => _setLive(v))),
            IconButton(
              tooltip: 'Refresh now',
              icon: const Icon(Icons.refresh),
              onPressed: _refresh,
            ),
            if (j.running)
              IconButton(
                tooltip: 'Stop job',
                icon: const Icon(
                  Icons.stop_circle_outlined,
                  color: Colors.redAccent,
                ),
                onPressed: () => widget.runner.stop(j),
              ),
          ],
        ),
        Expanded(
          child: Container(
            color: const Color(0xFF101418),
            padding: const EdgeInsets.all(8),
            child: SingleChildScrollView(
              controller: scroll,
              child: SelectableText(
                text.isEmpty ? '(no output yet)' : text,
                style: const TextStyle(
                  fontFamily: 'monospace',
                  fontSize: 12,
                  color: Color(0xFFD6DEE6),
                ),
              ),
            ),
          ),
        ),
      ],
    );
  }
}

void showLogDialog(BuildContext context, JobRunner runner, Job job) {
  showDialog(
    context: context,
    builder: (_) => Dialog(
      child: SizedBox(
        width: 900,
        height: 600,
        child: Padding(
          padding: const EdgeInsets.all(12),
          child: LogViewer(runner: runner, job: job),
        ),
      ),
    ),
  );
}
