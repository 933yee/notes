---
title: Flutter 筆記
date: 2024-05-03 23:37:31
tags: Flutter
category:
math: true
---

```dart
import 'package:flutter/material.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: Scaffold(
          backgroundColor: Colors.deepPurple[200],
          appBar: AppBar(
              title: const Text("Hello World"),
              backgroundColor: Colors.deepPurple[400],
              elevation: 0,
              leading: IconButton(
                icon: const Icon(Icons.menu),
                onPressed: () {},
              )),
          body: Center(
              child: Container(
            height: 500,
            width: 300,
            decoration: BoxDecoration(
              color: Colors.deepPurple,
              borderRadius: BorderRadius.circular(20),
            ),
            padding: const EdgeInsets.all(20),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                const Text(
                  "Hello World",
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 30,
                  ),
                ),
                Expanded(
                  child: Container(
                    color: Colors.deepPurple[100],
                  ),
                ),
                Expanded(
                  flex: 2,
                  child: Container(
                    color: Colors.deepPurple[200],
                  ),
                ),
                Expanded(
                  child: Container(
                    color: Colors.deepPurple[300],
                  ),
                )
              ],
            ),
          ))),
    );
  }
}
```

![demo 1](./images/flutter-notes/demo1.png)

### ListView and GridView

```dart
class MyApp extends StatelessWidget {
  MyApp({Key? key}) : super(key: key);
  final List names = [
    'John',
    'Doe',
    'Smith',
    'Alex',
    'James',
    'Robert',
    'William',
    'David',
    'Richard',
    'Joseph'
  ];
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
        debugShowCheckedModeBanner: false,
        home: Scaffold(
            body: Column(
          children: [
            Expanded(
              child: ListView.builder(
                  itemCount: 10,
                  itemBuilder: (context, index) => ListTile(
                        title: Text('Item $index'),
                      )),
            ),
            Expanded(
              flex: 2,
              child: GridView.builder(
                itemCount: 64,
                gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
                    crossAxisCount: 8),
                itemBuilder: (count, index) => Container(
                    color: Colors.deepPurple, margin: EdgeInsets.all(2)),
              ),
            ),
            Expanded(
              child: ListView.builder(
                  itemCount: names.length,
                  itemBuilder: (context, index) =>
                      ListTile(title: Text(names[index]))),
            ),
          ],
        )));
  }
}
```

![demo 2](./images/flutter-notes/demo2.png)

### Stack and GestureDetector

```dart
class MyApp extends StatelessWidget {
  MyApp({Key? key}) : super(key: key);

  void userTapped() {
    print("User tapped the container");
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
        debugShowCheckedModeBanner: false,
        home: Scaffold(
          body: Center(
            child: Stack(
              alignment: Alignment.center,
              children: [
                Container(
                  height: 300,
                  width: 300,
                  color: Colors.deepPurple,
                ),
                Container(
                  height: 200,
                  width: 200,
                  color: Colors.deepPurple[400],
                ),
                GestureDetector(
                  onTap: userTapped,
                  child: Container(
                    height: 100,
                    width: 100,
                    color: Colors.deepPurple[200],
                    child: Center(child: Text("Tap me!")),
                  ),
                )
              ],
            ),
          ),
        ));
  }
}
```

![demo 3](./images/flutter-notes/demo3.png)

### Navigation

#### Main

```dart
void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: FirstPage(),
      routes: {
        '/second': (context) => const SecondPage(),
      },
    );
  }
}
```

#### First Page

```dart
class FirstPage extends StatelessWidget {
  const FirstPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('First Page')),
      body: Center(
          child: ElevatedButton(
        child: Text("Go to Second Page"),
        onPressed: () {
          Navigator.pushNamed(context, '/second');
        },
      )),
    );
  }
}
```

#### Second Page

```dart
class SecondPage extends StatelessWidget {
  const SecondPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Second Page')),
    );
  }
}
```

### Drawer And Navigation

#### Main

```dart
void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: FirstPage(),
      routes: {
        '/firstpage': (context) => FirstPage(),
        '/homepage': (context) => HomePage(),
        '/settingspage': (context) => SettingsPage(),
      },
    );
  }
}
```

#### First Page

```dart
class FirstPage extends StatelessWidget {
  const FirstPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('First Page')),
      drawer: Drawer(
          backgroundColor: Colors.deepPurple[100],
          child: Column(
            children: [
              DrawerHeader(
                  child: Icon(
                Icons.favorite,
                size: 48,
              )),
              ListTile(
                leading: Icon(Icons.home),
                title: const Text('Home'),
                onTap: () {
                  Navigator.pop(context);
                  Navigator.pushNamed(context, '/homepage');
                },
              ),
              ListTile(
                leading: Icon(Icons.settings),
                title: const Text('Settings'),
                onTap: () {
                  Navigator.pop(context);
                  Navigator.pushNamed(context, '/settingspage');
                },
              ),
            ],
          )),
    );
  }
}
```

#### Home

```dart
class HomePage extends StatelessWidget {
  const HomePage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Home Page')),
    );
  }
}
```

![demo 4](./images/flutter-notes/demo4.png)

### Bottom Navigation

#### Main

```dart
void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: FirstPage(),
      routes: {
        '/firstpage': (context) => FirstPage(),
        '/homepage': (context) => HomePage(),
        '/settingspage': (context) => SettingsPage(),
      },
    );
  }
}
```

#### First Page (StateFul)

```dart
class FirstPage extends StatefulWidget {
  FirstPage({super.key});

  @override
  State<FirstPage> createState() => _FirstPageState();
}

class _FirstPageState extends State<FirstPage> {
  int _selectedIndex = 0;

  void _navigateBottomBar(int index) {
    setState(() {
      _selectedIndex = index;
    });
  }

  final List _pages = [
    HomePage(),
    ProfilePage(),
    SettingsPage(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('First Page')),
      body: _pages[_selectedIndex],
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _selectedIndex,
        onTap: _navigateBottomBar,
        items: [
          BottomNavigationBarItem(icon: Icon(Icons.home), label: 'Home'),
          BottomNavigationBarItem(icon: Icon(Icons.person), label: 'Profile'),
          BottomNavigationBarItem(
              icon: Icon(Icons.settings), label: 'Settings'),
        ],
      ),
    );
  }
}
```

#### Hone

```dart
class HomePage extends StatelessWidget {
  const HomePage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(child: Text('Home Page')),
      backgroundColor: Colors.amber[100],
    );
  }
}
```

![demo 5](./images/flutter-notes/demo5.png)

### Counter App

#### Main

```dart
void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: CounterPage(),
    );
  }
}
```

#### Counter Page

```dart
class CounterPage extends StatefulWidget {
  const CounterPage({super.key});

  @override
  State<CounterPage> createState() => _CounterPageState();
}

class _CounterPageState extends State<CounterPage> {
  int _counter = 0;
  void _incrementCounter() {
    setState(() {
      _counter++;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text("You pushed the button this many times:"),
            Text(
              '$_counter',
              style: TextStyle(fontSize: 40),
            ),
            ElevatedButton(
                onPressed: _incrementCounter, child: Text("Increment"))
          ],
        ),
      ),
    );
  }
}
```

### TextField

#### Main

```dart
void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: ToDoPage(),
    );
  }
}
```

#### ToDoPage

```dart
class ToDoPage extends StatefulWidget {
  const ToDoPage({super.key});

  @override
  State<ToDoPage> createState() => _ToDoPageState();
}

class _ToDoPageState extends State<ToDoPage> {
  TextEditingController myController = TextEditingController();
  String greetingMessage = "";

  void greetUser() {
    setState(() {
      greetingMessage = "Hello, ${myController.text}";
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(25.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text(greetingMessage),
              TextField(
                controller: myController,
                decoration: InputDecoration(
                  border: OutlineInputBorder(),
                  labelText: "Enter your name",
                ),
              ),
              ElevatedButton(onPressed: greetUser, child: Text("Tap"))
            ],
          ),
        ),
      ),
    );
  }
}
```

![demo 6](./images/flutter-notes/demo6.png)
