import React, { createContext, useContext, useMemo, useState } from 'react';
import { lightColors, darkColors, TEXT_SIZES } from './theme';

const SettingsContext = createContext(null);

export function SettingsProvider({ children }) {
  const [isDark, setIsDark] = useState(false);
  const [textSize, setTextSize] = useState('medium');

  const value = useMemo(() => {
    const scale = TEXT_SIZES[textSize];
    return {
      isDark,
      setIsDark,
      textSize,
      setTextSize,
      colors: isDark ? darkColors : lightColors,
      fs: (n) => Math.round(n * scale), // scaled font size
    };
  }, [isDark, textSize]);

  return (
    <SettingsContext.Provider value={value}>{children}</SettingsContext.Provider>
  );
}

export const useSettings = () => useContext(SettingsContext);