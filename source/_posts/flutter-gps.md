---
date: 2024-01-25 22:31:12
title: Flutter GPS background
category: [Flutter]
tags: [Flutter, App, ECHO]
---

> 官方文件: https://pub.dev/packages/background_locator_2/example
> Github: https://github.com/Yukams/background_locator_fixed

### 介紹

這個 package 的目的是讓 app 可以順利定位，取得定位的相關訊息 (經緯度、方向、速度等)，最重要的是它也支援**背景運行**，也就是當你 kill 這個 app (在 app switcher 裡 swipe up app)，這個 app 依然在背景中運行，並顯示在手機的 notification drawer 裡來提醒使用者。

![Notification](/images/flutter-gps/notification.png)

由於 `location_permissions` 似乎不再更新了，所以在測試的時候都會有問題，無法正確 import package，所以後來我改用 `permission_handler`，除了詢問 location 的權限，同時也可以要求 notification 的權限。

### 程式碼

```dart
import 'dart:async';
import 'dart:isolate';
import 'dart:ui';

import 'package:background_locator_2/background_locator.dart';
import 'package:background_locator_2/location_dto.dart';
import 'package:background_locator_2/settings/android_settings.dart';
import 'package:background_locator_2/settings/ios_settings.dart';
import 'package:background_locator_2/settings/locator_settings.dart';
import 'package:flutter/material.dart';

import 'package:permission_handler/permission_handler.dart';

import 'file_manager.dart';
import 'location_callback_handler.dart';
import 'location_service_repository.dart';

void main() => runApp(MyApp());

class MyApp extends StatefulWidget {
  @override
  _MyAppState createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  ReceivePort port = ReceivePort();

  String logStr = '';
  bool isRunning = false;

  @override
  void initState() {
    super.initState();

    if (IsolateNameServer.lookupPortByName(LocationServiceRepository.isolateName) != null)
    {
          IsolateNameServer.removePortNameMapping(
          LocationServiceRepository.isolateName);
    }

    IsolateNameServer.registerPortWithName(port.sendPort, LocationServiceRepository.isolateName);

    port.listen((dynamic data) async {
        await updateUI(data);
      },
    );
    initPlatformState();
  }

  @override
  void dispose() {
    super.dispose();
  }

  Future<void> updateUI(dynamic data) async {
    final log = await FileManager.readLogFile();

    if (data != null){
      await _updateNotificationText(data);
    }

    setState((){
      logStr = log;
    });
  }

  Future<void> _updateNotificationText(Map<String, dynamic> data) async {
    await BackgroundLocator.updateNotificationText(
        title: "new location received",
        msg: "${DateTime.now()}",
        bigMsg: "${data['latitude']}, ${data['longitude']}"
    );
  }

  Future<void> initPlatformState() async {
    print('Initializing...');
    await BackgroundLocator.initialize();
    logStr = await FileManager.readLogFile();
    print('Initialization done');
    final _isRunning = await BackgroundLocator.isServiceRunning();
    setState(() {
      isRunning = _isRunning;
    });
    print('Running ${isRunning.toString()}');
  }

  @override
  Widget build(BuildContext context) {
    final start = SizedBox(
      width: double.maxFinite,
      child: ElevatedButton(
        child: Text('Start'),
        onPressed: () {
          _onStart();
        },
      ),
    );
    final stop = SizedBox(
      width: double.maxFinite,
      child: ElevatedButton(
        child: Text('Stop'),
        onPressed: () {
          onStop();
        },
      ),
    );
    final clear = SizedBox(
      width: double.maxFinite,
      child: ElevatedButton(
        child: Text('Clear Log'),
        onPressed: () {
          FileManager.clearLogFile();
          setState(() {
            logStr = '';
          });
        },
      ),
    );
    String msgStatus = "-";
    if (isRunning != null) {
      if (isRunning) {
        msgStatus = 'Is running';
      } else {
        msgStatus = 'Is not running';
      }
    }
    final status = Text("Status: $msgStatus");

    final log = Text(
      logStr,
    );

    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          title: const Text('Flutter background Locator'),
        ),
        body: Container(
          width: double.maxFinite,
          padding: const EdgeInsets.all(22),
          child: SingleChildScrollView(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.center,
              children: <Widget>[start, stop, clear, status, log],
            ),
          ),
        ),
      ),
    );
  }

  void onStop() async {
    await BackgroundLocator.unRegisterLocationUpdate();
    final _isRunning = await BackgroundLocator.isServiceRunning();
    setState(() {
      isRunning = _isRunning;
    });
  }

  void _onStart() async {
    if (!await _checkNotificationPermission()) return;
    if(!await _checkLocationPermission()) return;
    await _startLocator();
    final _isRunning = await BackgroundLocator.isServiceRunning();

    setState(() {
      isRunning = _isRunning;
    });
  }

  Future<bool> _checkLocationPermission() async {
    Permission _permission = Permission.location;
    PermissionStatus _status = await _permission.request();
    if (_status.isPermanentlyDenied) {
      await openAppSettings();
    }
    return _status.isGranted;
  }

  Future<bool> _checkNotificationPermission() async {
    Permission _permission = Permission.notification;
    PermissionStatus _status = await _permission.request();
    if (_status.isPermanentlyDenied) {
      await openAppSettings();
    }
    return _status.isGranted;
  }

  Future<void> _startLocator() async{

    Map<String, dynamic> data = {'countInit': 1};
    return await BackgroundLocator.registerLocationUpdate(
        LocationCallbackHandler.callback,
        initCallback: LocationCallbackHandler.initCallback,
        initDataCallback: data,
        disposeCallback: LocationCallbackHandler.disposeCallback,
        iosSettings: IOSSettings(
            accuracy: LocationAccuracy.NAVIGATION,
            distanceFilter: 0,
            stopWithTerminate: true
        ),
        autoStop: false,
        androidSettings: AndroidSettings(
            accuracy: LocationAccuracy.NAVIGATION,
            interval: 1,
            distanceFilter: 0,
            client: LocationClient.google,
            androidNotificationSettings: AndroidNotificationSettings(
                notificationChannelName: 'Location tracking',
                notificationTitle: 'Start Location Tracking',
                notificationMsg: 'Track location in background',
                notificationBigMsg:
                'Background location is on to keep the app up-tp-date with your location. This is required for main features to work properly when the app is not running.',
                notificationIcon: '',
                notificationIconColor: Colors.grey,
                notificationTapCallback:
                LocationCallbackHandler.notificationCallback)
        )
    );
  }
}
```
在 AndroidMaifest.xml 裡面要加上：
```xml
<uses-permission android:name="android.permission.POST_NOTIFICATIONS"/>
<uses-permission android:name="android.permission.ACCESS_FINE_LOCATION"/>
<uses-permission android:name="android.permission.ACCESS_BACKGROUND_LOCATION" />
<uses-permission android:name="android.permission.WAKE_LOCK"/>
<uses-permission android:name="android.permission.FOREGROUND_SERVICE"/>
```

### Demo
[![](https://markdown-videos-api.jorgenkh.no/youtube/euh3HlNAERs)](https://youtu.be/euh3HlNAERs)