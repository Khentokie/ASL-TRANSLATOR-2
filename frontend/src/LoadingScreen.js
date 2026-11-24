import React, { useEffect, useRef } from 'react';
import { View, Text, Animated, StyleSheet } from 'react-native';
import { StatusBar } from 'expo-status-bar';
import Logo from './Logo';

export default function LoadingScreen({ fontsReady }) {
  const fade = useRef(new Animated.Value(0)).current;
  const scale = useRef(new Animated.Value(0.85)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.timing(fade, { toValue: 1, duration: 700, useNativeDriver: true }),
      Animated.spring(scale, { toValue: 1, friction: 6, useNativeDriver: true }),
    ]).start();
  }, []);

  return (
    <View style={styles.container}>
      <StatusBar style="light" />
      <Animated.View style={{ alignItems: 'center', opacity: fade, transform: [{ scale }] }}>
        <Logo size={200} />
        <Text style={[styles.name, fontsReady && { fontFamily: 'Outfit_700Bold' }]}>
          Signify
        </Text>
        <Text style={[styles.tag, fontsReady && { fontFamily: 'Outfit_600SemiBold' }]}>
          AI-ASSISTED COMMUNICATION
        </Text>
      </Animated.View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#035FF2',
    alignItems: 'center',
    justifyContent: 'center',
  },
  name: { fontSize: 42, fontWeight: '700', color: '#fff', marginTop: 8 },
  tag: { fontSize: 12, letterSpacing: 2, color: 'rgba(255,255,255,0.8)', marginTop: 6 },
});