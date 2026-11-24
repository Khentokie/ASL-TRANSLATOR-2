import React from 'react';
import { Pressable } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import { useSettings } from './SettingsContext';

export default function SettingsButton() {
  const navigation = useNavigation();
  const { colors } = useSettings();

  return (
    <Pressable
      onPress={() => navigation.navigate('Settings')}
      hitSlop={10}
      style={({ pressed }) => ({
        width: 44,
        height: 44,
        borderRadius: 14,
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: colors.card,
        opacity: pressed ? 0.7 : 1,
      })}
    >
      <Ionicons name="settings-outline" size={22} color={colors.text} />
    </Pressable>
  );
}