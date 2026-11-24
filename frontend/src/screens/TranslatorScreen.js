import React, { useState, useMemo } from 'react';
import { View, Text, Pressable, StyleSheet, ScrollView, Alert } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as Clipboard from 'expo-clipboard';
import { Ionicons, Feather } from '@expo/vector-icons';
import { fonts, panel } from '../theme';
import { useSettings } from '../SettingsContext';
import SettingsButton from '../SettingsButton';

// Fake AI output for the prototype
const DEMO = {
  spelling: { subtitle: 'Finger spelling to text', big: 'HELLO', sentence: 'Nice to meet you.' },
  word: { subtitle: 'ASL word to text', big: 'THANK YOU', sentence: 'Thank you very much.' },
};

export default function TranslatorScreen({ navigation }) {
  const { colors, fs } = useSettings();
  const styles = useMemo(() => makeStyles(colors, fs), [colors, fs]);

  const [mode, setMode] = useState('spelling');
  const [cleared, setCleared] = useState(false);
  const [permission, requestPermission] = useCameraPermissions();

  const data = DEMO[mode];
  const fullText = cleared ? '' : `${data.big} ${data.sentence}`;
  const wordCount = fullText.trim() ? fullText.trim().split(/\s+/).length : 0;

  const onCopy = async () => {
    await Clipboard.setStringAsync(fullText);
    Alert.alert('Copied', 'Translation copied to clipboard.');
  };
  const switchMode = (m) => { setMode(m); setCleared(false); };

  return (
    <SafeAreaView style={styles.container}>
      {/* Header: Back (left) + Settings (right) */}
      <View style={styles.header}>
        <Pressable style={styles.back} onPress={() => navigation.goBack()}>
          <Feather name="arrow-left" size={20} color={colors.muted} />
          <Text style={styles.backText}>Back</Text>
        </Pressable>
        <SettingsButton />
      </View>

      <ScrollView contentContainerStyle={{ paddingBottom: 24 }}>
        {/* Camera panel */}
        <View style={styles.panel}>
          <View style={styles.cameraBox}>
            {permission?.granted ? (
              <CameraView style={StyleSheet.absoluteFill} facing="front" />
            ) : (
              <Pressable style={styles.center} onPress={requestPermission}>
                <Ionicons name="hand-left-outline" size={110} color="#5B8DFF" />
                <Text style={styles.permText}>Tap to enable camera</Text>
              </Pressable>
            )}

            <View style={styles.frame} pointerEvents="none" />

            <View style={styles.badge}>
              <Text style={styles.badgeText}>HAND DETECTED · 96%</Text>
            </View>
          </View>

          {/* Mode toggle */}
          <View style={styles.toggle}>
            <ToggleBtn
              label="Finger Spelling"
              active={mode === 'spelling'}
              onPress={() => switchMode('spelling')}
              styles={styles}
            />
            <ToggleBtn
              label="ASL Word"
              active={mode === 'word'}
              onPress={() => switchMode('word')}
              styles={styles}
            />
          </View>
        </View>

        {/* Result card */}
        <View style={styles.card}>
          <Text style={styles.cardLabel}>TRANSLATED TEXT</Text>
          <Text style={styles.cardSub}>{data.subtitle}</Text>

          {!cleared && (
            <>
              <Text style={styles.big}>{data.big.split('').join(' ')}</Text>
              <Text style={styles.sentence}>{data.sentence}</Text>
            </>
          )}

          <View style={styles.meta}>
            <Text style={styles.metaText}>{wordCount} words</Text>
            <Text style={styles.metaText}>96% confidence</Text>
          </View>

          <View style={styles.actions}>
            <Pressable onPress={() => setCleared(true)} hitSlop={10}>
              <Feather name="trash-2" size={20} color={colors.muted} />
            </Pressable>
            <Pressable style={styles.action} onPress={onCopy}>
              <Feather name="copy" size={20} color={colors.muted} />
              <Text style={styles.actionText}>Copy</Text>
            </Pressable>
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

function ToggleBtn({ label, active, onPress, styles }) {
  return (
    <Pressable onPress={onPress} style={[styles.toggleBtn, active && styles.toggleActive]}>
      <Ionicons name="hand-left-outline" size={16} color="#fff" />
      <Text style={styles.toggleText}>{label}</Text>
    </Pressable>
  );
}

const makeStyles = (colors, fs) =>
  StyleSheet.create({
    container: { flex: 1, backgroundColor: colors.bg, paddingHorizontal: 16 },
    header: {
      flexDirection: 'row', justifyContent: 'space-between',
      alignItems: 'center', paddingVertical: 8,
    },
    back: { flexDirection: 'row', alignItems: 'center', gap: 6 },
    backText: { fontFamily: fonts.medium, fontSize: fs(16), color: colors.muted },

    panel: { backgroundColor: panel.navy, borderRadius: 28, overflow: 'hidden', marginTop: 8 },
    cameraBox: {
      height: 320, backgroundColor: panel.navyLight,
      justifyContent: 'center', alignItems: 'center',
    },
    center: { alignItems: 'center' },
    permText: { fontFamily: fonts.medium, fontSize: fs(15), color: '#9DB7FF', marginTop: 8 },
    frame: {
      position: 'absolute', top: 40, bottom: 40, left: 50, right: 50,
      borderWidth: 2, borderColor: 'rgba(91,141,255,0.45)', borderRadius: 28,
    },
    badge: {
      position: 'absolute', top: 14, alignSelf: 'center',
      backgroundColor: '#1E6BFF', paddingHorizontal: 14, paddingVertical: 6, borderRadius: 16,
    },
    badgeText: { fontFamily: fonts.bold, fontSize: fs(11), letterSpacing: 0.8, color: '#fff' },

    toggle: {
      flexDirection: 'row', backgroundColor: panel.toggleBg,
      margin: 12, padding: 6, borderRadius: 20,
    },
    toggleBtn: {
      flex: 1, height: 48, borderRadius: 16, flexDirection: 'row',
      alignItems: 'center', justifyContent: 'center', gap: 8,
    },
    toggleActive: { backgroundColor: '#1E6BFF' },
    toggleText: { fontFamily: fonts.semi, fontSize: fs(14), color: '#fff' },

    card: { backgroundColor: colors.card, borderRadius: 24, padding: 18, marginTop: 16 },
    cardLabel: {
      fontFamily: fonts.bold, fontSize: fs(12), letterSpacing: 1.5, color: colors.primary,
    },
    cardSub: { fontFamily: fonts.regular, fontSize: fs(18), color: colors.text, marginTop: 4 },
    big: {
      fontFamily: fonts.medium, fontSize: fs(20), letterSpacing: 4,
      color: colors.muted, marginTop: 24,
    },
    sentence: { fontFamily: fonts.regular, fontSize: fs(18), color: colors.text, marginTop: 24 },
    meta: { flexDirection: 'row', justifyContent: 'space-between', marginTop: 32 },
    metaText: { fontFamily: fonts.regular, fontSize: fs(12), color: colors.muted },
    actions: {
      flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
      marginTop: 16, paddingHorizontal: 8,
    },
    action: { flexDirection: 'row', alignItems: 'center', gap: 6 },
    actionText: { fontFamily: fonts.medium, fontSize: fs(15), color: colors.muted },
  });