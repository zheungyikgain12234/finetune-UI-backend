import 'package:flutter/material.dart';

import '../services/job_runner.dart';
import '../widgets/common.dart';

class _AgentPreset {
  const _AgentPreset(this.name, this.url);
  final String name, url;
}

const _agentPresets = [
  _AgentPreset('byoa-md-json (public)', 'https://md-json.zackloh.dev'),
  _AgentPreset('byoa-md-json (this machine)', 'http://localhost:5700'),
  _AgentPreset('Custom agent', ''),
];

/// "Agent" mode of the teacher: send each text/markdown file to an agent endpoint and save its JSON,
/// keeping only the fields the user names.
class AgentForm extends StatefulWidget {
  const AgentForm({super.key, required this.runner});
  final JobRunner runner;
  @override
  State<AgentForm> createState() => _AgentFormState();
}

class _AgentFormState extends State<AgentForm> {
  _AgentPreset preset = _agentPresets[1];
  final url = TextEditingController(text: _agentPresets[1].url);
  final secret = TextEditingController();
  final capability = TextEditingController(text: 'text-extract');
  final contentField = TextEditingController(text: 'md');
  final fields = TextEditingController();
  final instructions = TextEditingController();
  final input = TextEditingController();
  final output = TextEditingController();
  final workers = TextEditingController(text: '2');
  final timeout = TextEditingController(text: '600');
  Map<String, dynamic>? card;
  bool testing = false;
  String? msg;

  JobRunner get r => widget.runner;

  Future<void> test() async {
    setState(() {
      testing = true;
      card = null;
    });
    Map<String, dynamic> c;
    try {
      c = await r.agentCard(url.text.trim());
    } catch (e) {
      c = {'ok': false, 'error': '$e'};
    }
    if (!mounted) return;
    setState(() {
      testing = false;
      card = c;
      if (c['ok'] == true) {
        capability.text = c['capability'] ?? capability.text;
        final ins = (c['inputs'] as Map?) ?? {};
        final firstReq = ins.entries.firstWhere(
          (e) =>
              e.value is Map &&
              e.value['required'] == true &&
              e.value['type'] == 'string',
          orElse: () => const MapEntry('md', null),
        );
        contentField.text = firstReq.key;
      }
    });
  }

  Future<void> run() async {
    if ([
      url,
      secret,
      input,
      output,
      capability,
      contentField,
    ].any((c) => c.text.trim().isEmpty)) {
      setState(
        () => msg = 'Fill in the agent URL, secret, capability, input folder and output folder.',
      );
      return;
    }
    final cmd =
        'python3 ${shq('${r.root}/ovis_ft_ui/scripts/teacher_agent.py')} ${shq(input.text.trim())} '
        '--out ${shq(output.text.trim())} --url ${shq(url.text.trim())} '
        '--capability ${shq(capability.text.trim())} --content-field ${shq(contentField.text.trim())} '
        '--workers ${int.tryParse(workers.text) ?? 2} --timeout ${int.tryParse(timeout.text) ?? 600} '
        '--fields ${shq(fields.text.trim())} --instructions ${shq(instructions.text.trim())}';
    final job = await r.start(
      'teacher',
      cmd,
      env: {'AGENT_SECRET': secret.text.trim()},
    );
    setState(() => msg = 'Started in background (job ${job.id}).');
  }

  @override
  Widget build(BuildContext context) {
    final ok = card?['ok'] == true;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Section(
          '1. Agent',
          children: [
            DropdownButtonFormField<_AgentPreset>(
              initialValue: preset,
              decoration: const InputDecoration(
                labelText: 'Agent',
                border: OutlineInputBorder(),
              ),
              items: [
                for (final p in _agentPresets)
                  DropdownMenuItem(value: p, child: Text(p.name)),
              ],
              onChanged: (p) => setState(() {
                preset = p!;
                if (p.url.isNotEmpty) url.text = p.url;
                card = null;
              }),
            ),
            gap(),
            TextField(
              controller: url,
              decoration: const InputDecoration(
                labelText: 'Agent endpoint URL (the server calls it, so "localhost" means the gx10)',
                border: OutlineInputBorder(),
              ),
            ),
            gap(),
            TextField(
              controller: secret,
              obscureText: true,
              decoration: const InputDecoration(
                labelText: 'Agent secret (BYOA_SECRET, used to sign requests)',
                helperText:
                    'Passed to the job via environment only; never saved.',
                border: OutlineInputBorder(),
              ),
            ),
            gap(),
            Row(
              children: [
                FilledButton.icon(
                  onPressed: testing ? null : test,
                  icon: testing
                      ? const SizedBox(
                          width: 16,
                          height: 16,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Icon(Icons.wifi_tethering),
                  label: const Text('Test agent (health + card)'),
                ),
                gap(),
                if (card != null)
                  Expanded(
                    child: Text(
                      ok
                          ? '✓ ${card!['name']} v${card!['version']}: ${card!['description'] ?? ''}'
                          : '✗ ${card!['error'] ?? 'unreachable'}',
                      style: TextStyle(
                        color: ok ? Colors.green : Colors.redAccent,
                      ),
                    ),
                  ),
              ],
            ),
            if (ok)
              Padding(
                padding: const EdgeInsets.only(top: 6),
                child: Text(
                  'Capability: ${card!['capability']}   •   inputs: ${(card!['inputs'] as Map).keys.join(', ')}',
                  style: Theme.of(context).textTheme.bodySmall,
                ),
              ),
            gap(),
            Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: capability,
                    decoration: const InputDecoration(
                      labelText: 'Capability',
                      border: OutlineInputBorder(),
                    ),
                  ),
                ),
                gap(),
                Expanded(
                  child: TextField(
                    controller: contentField,
                    decoration: const InputDecoration(
                      labelText: 'Input field for the file text',
                      border: OutlineInputBorder(),
                    ),
                  ),
                ),
              ],
            ),
          ],
        ),
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
                    '(only the extracted JSON),  response.duration_ms,  response.outputs.structured.total_sgd.  '
                    'A bare name like  document_number, line_items.description  is searched inside it and also '
                    'sent to the agent as what to extract.',
                helperMaxLines: 5,
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
                labelText: 'Extra instructions for the agent (optional)',
                helperText: 'Sent as the agent\'s "focus" together with any bare field names.',
                helperMaxLines: 2,
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
              label:
                  'Input folder (markdown / text files, searched recursively)',
              controller: input,
            ),
            gap(),
            FolderField(
              runner: r,
              label: 'Output folder (one .json per input; raw agent replies in _raw/)',
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
                        controller: workers,
                        decoration: const InputDecoration(
                          labelText: 'Parallel workers',
                          border: OutlineInputBorder(),
                        ),
                      ),
                    ),
                    gap(),
                    Expanded(
                      child: TextField(
                        controller: timeout,
                        decoration: const InputDecoration(
                          labelText: 'Timeout per file (s)',
                          border: OutlineInputBorder(),
                        ),
                      ),
                    ),
                  ],
                ),
              ],
            ),
            gap(),
            Row(
              children: [
                FilledButton.icon(
                  onPressed: r.isRunning('teacher') ? null : run,
                  icon: const Icon(Icons.play_arrow),
                  label: const Text('Run agent in background'),
                ),
                gap(),
                if (msg != null) Expanded(child: Text(msg!)),
              ],
            ),
            const SizedBox(height: 4),
            const Text(
              'Finished files in the output folder are skipped, so re-running resumes.',
              style: TextStyle(fontSize: 12),
            ),
          ],
        ),
      ],
    );
  }
}
