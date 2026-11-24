import React, { useMemo } from 'react';
import { View, Text, Pressable, Switch, StyleSheet } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Feather, Ionicons } from '@expo/vector-icons';
import { fonts } from '../theme';
import { useSettings } from '../SettingsContext';

const SIZES = [
  { key: 'small', label: 'Small', preview: 14 },
  { key: 'medium', label: 'Medium', preview: 18 },
  { key: 'large', label: 'Large', preview: 24 },
];

export default function SettingsScreen({ navigation }) {
  const { colors, fs, isDark, setIsDark, textSize, setTextSize } = useSettings();
  const styles = useMemo(() => makeStyles(colors, fs), [colors, fs]);

  return (
    <SafeAreaView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Pressable style={styles.back} onPress={() => navigation.goBack()}>
          <Feather name="arrow-left" size={20} color={colors.muted} />
          <Text style={styles.backText}>Back</Text>
        </Pressable>
        <Text style={styles.headerTitle}>Settings</Text>
        <View style={{ width: 60 }} />
      </View>

      {/* Dark mode */}
      <View style={styles.card}>
        <View style={styles.row}>
          <View style={styles.rowLeft}>
            <Ionicons name={isDark ? 'moon' : 'moon-outline'} size={22} color={colors.primary} />
            <Text style={styles.rowTitle}>Dark mode</Text>
          </View>
          <Switch
            value={isDark}
            onValueChange={setIsDark}
            trackColor={{ false: '#C9CFE8', true: colors.primary }}
            thumbColor="#fff"
          />
        </View>
      </View>

      {/* Text size */}
      <View style={styles.card}>
        <View style={styles.rowLeft}>
          <Ionicons name="text-outline" size={22} color={colors.primary} />
          <Text style={styles.rowTitle}>Text size</Text>
        </View>

        <View style={styles.sizeRow}>
          {SIZES.map((s) => {
            const active = textSize === s.key;
            return (
              <Pressable
                key={s.key}
                onPress={() => setTextSize(s.key)}
                style={[styles.sizeBtn, active && styles.sizeBtnActive]}
              >
                <Text
                  style={[
                    styles.sizeAa,
                    { fontSize: s.preview },
                    active && { color: '#fff' },
                  ]}
                >
                  Aa
                </Text>
                <Text style={[styles.sizeLabel, active && { color: '#fff' }]}>
                  {s.label}
                </Text>
              </Pressable>
            );
          })}
        </View>

        <Text style={styles.preview}>
          Preview: Every sign deserves to be understood.
        </Text>
      </View>
    </SafeAreaView>
  );
}

const makeStyles = (colors, fs) =>
  StyleSheet.create({
    container: { flex: 1, backgroundColor: colors.bg, paddingHorizontal: 16 },
    header: {
      flexDirection: 'row', justifyContent: 'space-between',
      alignItems: 'center', paddingVertical: 8, marginBottom: 12,
    },
    back: { flexDirection: 'row', alignItems: 'center', gap: 6, width: 60 },
    backText: { fontFamily: fonts.medium, fontSize: fs(16), color: colors.muted },
    headerTitle: { fontFamily: fonts.semi, fontSize: fs(18), color: colors.text },

    card: { backgroundColor: colors.card, borderRadius: 24, padding: 18, marginBottom: 16 },
    row: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
    rowLeft: { flexDirection: 'row', alignItems: 'center', gap: 12 },
    rowTitle: { fontFamily: fonts.semi, fontSize: fs(17), color: colors.text },

    sizeRow: { flexDirection: 'row', gap: 10, marginTop: 16 },
    sizeBtn: {
      flex: 1, alignItems: 'center', paddingVertical: 14,
      borderRadius: 16, backgroundColor: colors.chip,
    },
    sizeBtnActive: { backgroundColor: colors.primary },
    sizeAa: { fontFamily: fonts.bold, color: colors.text },
    sizeLabel: { fontFamily: fonts.medium, fontSize: 12, color: colors.muted, marginTop: 4 },

    preview: {
      fontFamily: fonts.regular, fontSize: fs(15), lineHeight: fs(22),
      color: colors.muted, marginTop: 16,
    },
  });