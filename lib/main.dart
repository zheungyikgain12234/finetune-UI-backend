import 'package:flutter/material.dart';

import 'screens/student_screen.dart';
import 'screens/teacher_screen.dart';
import 'services/job_runner.dart';

void main() => runApp(const OvisFtApp());

class OvisFtApp extends StatelessWidget {
  const OvisFtApp({super.key});
  @override
  Widget build(BuildContext context) => MaterialApp(
    title: 'Ovis Fine-tune',
    theme: ThemeData(colorSchemeSeed: Colors.indigo, useMaterial3: true),
    darkTheme: ThemeData(
      colorSchemeSeed: Colors.indigo,
      brightness: Brightness.dark,
      useMaterial3: true,
    ),
    home: const Home(),
  );
}

class Home extends StatefulWidget {
  const Home({super.key});
  @override
  State<Home> createState() => _HomeState();
}

class _HomeState extends State<Home> {
  final runner = JobRunner();
  int index = 0;

  @override
  Widget build(BuildContext context) {
    final screens = [
      TeacherScreen(runner: runner),
      StudentScreen(runner: runner),
    ];
    return Scaffold(
      appBar: AppBar(
        title: const Text('Ovis fine-tune'),
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 16),
            child: Center(
              child: ListenableBuilder(
                listenable: runner,
                builder: (_, __) => Text(
                  'workspace: ${runner.root}',
                  style: const TextStyle(fontSize: 12),
                ),
              ),
            ),
          ),
        ],
      ),
      body: AnimatedBuilder(
        animation: runner,
        builder: (context, _) => Column(
          children: [
            if (runner.error != null)
              MaterialBanner(
                content: Text(runner.error!),
                backgroundColor: Theme.of(context).colorScheme.errorContainer,
                actions: [
                  TextButton(
                    onPressed: runner.refresh,
                    child: const Text('Retry'),
                  ),
                ],
              ),
            Expanded(
              child: Row(
                children: [
                  NavigationRail(
                    selectedIndex: index,
                    labelType: NavigationRailLabelType.all,
                    onDestinationSelected: (i) => setState(() => index = i),
                    destinations: const [
                      NavigationRailDestination(
                        icon: Icon(Icons.school_outlined),
                        label: Text('Teacher'),
                      ),
                      NavigationRailDestination(
                        icon: Icon(Icons.model_training),
                        label: Text('Student'),
                      ),
                    ],
                  ),
                  const VerticalDivider(width: 1),
                  Expanded(
                    child: IndexedStack(index: index, children: screens),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
