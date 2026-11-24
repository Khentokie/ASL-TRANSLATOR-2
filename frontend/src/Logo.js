import React from 'react';
import { Image } from 'react-native';

export default function Logo({ size = 48 }) {
  return (
    <Image
      source={require('../assets/logo-white.png')}
      style={{ width: size, height: size }}
      resizeMode="contain"
    />
  );
}