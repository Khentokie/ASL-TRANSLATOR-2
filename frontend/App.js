import React, { useEffect, useState } from 'react';
import { StatusBar } from 'expo-status-bar';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import {
  useFonts,
  Outfit_400Regular,
  Outfit_500Medium,
  Outfit_600SemiBold,
  Outfit_700Bold,
} from '@expo-google-fonts/outfit';

import { SettingsProvider, useSettings } from './src/SettingsContext';
import LoadingScreen from './src/LoadingScreen';
import WelcomeScreen from './src/screens/WelcomeScreen';
import TranslatorScreen from './src/screens/TranslatorScreen';
import SettingsScreen from './src/screens/SettingsScreen';

const Stack = createNativeStackNavigator();

function AppNav() {
  const { colors, isDark } = useSettings();
  return (
    <NavigationContainer>
      <StatusBar style={isDark ? 'light' : 'dark'} />
      <Stack.Navigator
        screenOptions={{
          headerShown: false,
          contentStyle: { backgroundColor: colors.bg },
        }}
      >
        <Stack.Screen name="Welcome" component={WelcomeScreen} />
        <Stack.Screen name="Translator" component={TranslatorScreen} />
        <Stack.Screen name="Settings" component={SettingsScreen} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}

export default function App() {
  const [fontsLoaded] = useFonts({
    Outfit_400Regular,
    Outfit_500Medium,
    Outfit_600SemiBold,
    Outfit_700Bold,
  });
  const [minTimePassed, setMinTimePassed] = useState(false);

  // Keep the loading screen visible for at least 2.2 seconds
  useEffect(() => {
    const t = setTimeout(() => setMinTimePassed(true), 2200);
    return () => clearTimeout(t);
  }, []);

  if (!fontsLoaded || !minTimePassed) {
    return <LoadingScreen fontsReady={fontsLoaded} />;
  }

  return (
    <SafeAreaProvider>
      <SettingsProvider>
        <AppNav />
      </SettingsProvider>
    </SafeAreaProvider>
  );
}