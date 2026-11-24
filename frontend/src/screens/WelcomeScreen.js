import React, { useMemo } from 'react';
import { View, Text, Pressable, StyleSheet } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Feather } from '@expo/vector-icons';
import { fonts } from '../theme';
import { useSettings } from '../SettingsContext';
import Logo from '../Logo';

export default function WelcomeScreen({ navigation }) {
  const { colors, fs } = useSettings();
  const styles = useMemo(() => makeStyles(colors, fs), [colors, fs]);

  return (
    <SafeAreaView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <View style={styles.logo}>
          <Logo size={46} />
        </View>
        <View style={{ flex: 1 }}>
          <Text style={styles.title}>Signify</Text>
          <Text style={styles.subtitle}>AI-ASSISTED COMMUNICATION</Text>
        </View>
  
      </View>

      {/* Hero card */}
      <LinearGradient colors={['#0A55F5', '#2B8CFF']} style={styles.hero}>
        <View style={styles.heroIconBox}>
          <Logo size={130} />
        </View>
        <View style={styles.pill}>
          <View style={styles.dot} />
          <Text style={styles.pillText}>Ready to translate</Text>
        </View>
      </LinearGradient>

      {/* Text */}
      <View style={styles.labelRow}>
        <View style={[styles.dot, { backgroundColor: colors.primary }]} />
        <Text style={styles.label}>REAL-TIME SIGN TRANSLATION</Text>
      </View>

      <Text style={styles.headline}>Every sign deserves</Text>
      <Text style={[styles.headline, { color: colors.primary }]}>
        to be understood.
      </Text>

      <Text style={styles.body}>
        Bridging communication between Deaf and hearing individuals through
        AI-assisted sign language translation.
      </Text>

      {/* Button */}
      <Pressable
        style={({ pressed }) => [styles.button, pressed && { opacity: 0.85 }]}
        onPress={() => navigation.navigate('Translator')}
      >
        <Text style={styles.buttonText}>Get Started</Text>
        <Feather name="arrow-right" size={18} color="#fff" />
      </Pressable>
    </SafeAreaView>
  );
}

const makeStyles = (colors, fs) =>
  StyleSheet.create({
    container: { flex: 1, backgroundColor: colors.bg, paddingHorizontal: 20 },
    header: { flexDirection: 'row', alignItems: 'center', gap: 12, marginTop: 8 },
    logo: {
      width: 48, height: 48, borderRadius: 14,
      backgroundColor: '#035FF2', alignItems: 'center', justifyContent: 'center',
      overflow: 'hidden',
    },
    title: { fontFamily: fonts.semi, fontSize: fs(20), color: colors.text },
    subtitle: {
      fontFamily: fonts.semi, fontSize: fs(11), letterSpacing: 1.5, color: colors.primary,
    },

    hero: {
      height: 260, borderRadius: 32, marginTop: 24,
      alignItems: 'center', justifyContent: 'center',
    },
    heroIconBox: {
      width: 150, height: 150, borderRadius: 36,
      backgroundColor: 'rgba(255,255,255,0.2)',
      alignItems: 'center', justifyContent: 'center',
    },
    pill: {
      position: 'absolute', right: 14, bottom: 14,
      flexDirection: 'row', alignItems: 'center', gap: 8,
      backgroundColor: '#fff', paddingHorizontal: 14, paddingVertical: 8, borderRadius: 20,
    },
    pillText: { fontFamily: fonts.medium, fontSize: fs(12), color: '#141B34' },
    dot: { width: 8, height: 8, borderRadius: 4, backgroundColor: colors.green },

    labelRow: {
      flexDirection: 'row', alignItems: 'center', justifyContent: 'center',
      gap: 8, marginTop: 28,
    },
    label: {
      fontFamily: fonts.semi, fontSize: fs(12), letterSpacing: 1.5, color: colors.primary,
    },

    headline: {
      fontFamily: fonts.bold, fontSize: fs(34), lineHeight: fs(42),
      textAlign: 'center', color: colors.text,
    },
    body: {
      fontFamily: fonts.regular, fontSize: fs(15), lineHeight: fs(24),
      textAlign: 'center', color: colors.muted, marginTop: 16, paddingHorizontal: 8,
    },
    button: {
      marginTop: 'auto', marginBottom: 12, height: 58, borderRadius: 18,
      backgroundColor: colors.primary, flexDirection: 'row',
      alignItems: 'center', justifyContent: 'center', gap: 10,
    },
    buttonText: { fontFamily: fonts.semi, fontSize: fs(17), color: '#fff' },
  });