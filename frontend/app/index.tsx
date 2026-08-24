import { useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Platform,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import { Link } from 'expo-router';
import {
  Inter_400Regular,
  Inter_500Medium,
  Inter_600SemiBold,
  Inter_700Bold,
  useFonts,
} from '@expo-google-fonts/inter';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import WebView from 'react-native-webview';

import { encryptEmail, type EncryptedSvdVector, type Prediction } from '@/lib/sealClient';
import exampleEmails from '@/assets/example-emails.json';

const TEST_EMAILS = exampleEmails.map((example, index) => ({
  id: String(index + 1),
  label: `Email ${index + 1} — ${example.classification}`,
  preview: example.email,
}));

type Tab = 'paste' | 'select';

export default function Index() {
  const [fontsLoaded, fontError] = useFonts({
    Inter_400Regular,
    Inter_500Medium,
    Inter_600SemiBold,
    Inter_700Bold,
  });
  const insets = useSafeAreaInsets();
  const [activeTab, setActiveTab] = useState<Tab>('select');
  const [pastedText, setPastedText] = useState('');
  const [selectedEmailId, setSelectedEmailId] = useState<string | null>(null);
  const [isEncrypting, setIsEncrypting] = useState(false);
  const [isClassifying, setIsClassifying] = useState(false);
  const [encryptingDots, setEncryptingDots] = useState(1);
  const [ciphertext, setCiphertext] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const encryptedSessionRef = useRef<EncryptedSvdVector | null>(null);
  const requestIdRef = useRef(0);

  const selectedEmail = TEST_EMAILS.find((email) => email.id === selectedEmailId);
  const emailText = activeTab === 'paste' ? pastedText : selectedEmail?.preview ?? '';
  const isBusy = isEncrypting || isClassifying;

  useEffect(() => {
    if (!isEncrypting) {
      return;
    }

    const interval = setInterval(() => {
      setEncryptingDots((dots) => (dots === 3 ? 1 : dots + 1));
    }, 350);
    return () => clearInterval(interval);
  }, [isEncrypting]);

  useEffect(() => {
    return () => encryptedSessionRef.current?.dispose();
  }, []);

  if (!fontsLoaded && !fontError) {
    return null;
  }

  function clearEncryption() {
    requestIdRef.current += 1;
    encryptedSessionRef.current?.dispose();
    encryptedSessionRef.current = null;
    setCiphertext(null);
    setPrediction(null);
  }

  function handleTabChange(tab: Tab) {
    if (isBusy) {
      return;
    }
    clearEncryption();
    setError(null);
    setActiveTab(tab);
  }

  function handlePasteChange(value: string) {
    clearEncryption();
    setError(null);
    setPastedText(value);
  }

  function handleEmailSelect(id: string) {
    if (isBusy) {
      return;
    }
    clearEncryption();
    setError(null);
    setSelectedEmailId(id);
  }

  async function handlePrimaryAction() {
    if (isClassifying || isEncrypting) {
      return;
    }

    const encryptedSession = encryptedSessionRef.current;
    if (encryptedSession) {
      setIsClassifying(true);
      setPrediction(null);
      setError(null);
      try {
        const result = await encryptedSession.classify();
        setPrediction(result);
        encryptedSession.dispose();
        encryptedSessionRef.current = null;
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : 'Encrypted prediction failed.');
      } finally {
        setIsClassifying(false);
      }
      return;
    }

    if (!emailText.trim()) {
      setError('Enter an email or select a test email first.');
      return;
    }
    if (Platform.OS !== 'web') {
      setError('The browser CKKS client is currently available on Expo web only.');
      return;
    }

    clearEncryption();
    setIsEncrypting(true);
    setEncryptingDots(1);
    setPrediction(null);
    setError(null);
    const requestId = ++requestIdRef.current;
    try {
      const session = await encryptEmail(emailText);
      if (requestId !== requestIdRef.current) {
        session.dispose();
        return;
      }
      encryptedSessionRef.current = session;
      setCiphertext(session.ciphertext);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Email encryption failed.');
    } finally {
      setIsEncrypting(false);
    }
  }

  return (
    <View style={[styles.container, { paddingTop: insets.top + 16 }]}>
      <Link href="/about" style={[styles.aboutLink, { top: insets.top + 12 }]}>
        About
      </Link>

      {/* Hidden WebView keeps the SEAL WASM context warm on mobile */}
      {Platform.OS !== 'web' && (
        <WebView
          style={styles.hiddenWebView}
          source={{ html: '<html><body></body></html>' }}
        />
      )}

      <View style={styles.header}>
        <Text style={styles.title}>Private Email Filter</Text>
        <Text style={styles.subtitle}>A homomorphic encryption demo</Text>
      </View>

      <View style={styles.card}>

        {/* Tab switcher */}
        <View style={styles.tabs}>
          <TouchableOpacity
            style={[styles.tab, styles.softShadow, activeTab === 'select' && styles.tabActive]}
            disabled={isBusy}
            onPress={() => handleTabChange('select')}
          >
            <Text style={[styles.tabText, activeTab === 'select' && styles.tabTextActive]}>
              Select email
            </Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.tab, styles.softShadow, activeTab === 'paste' && styles.tabActive]}
            disabled={isBusy}
            onPress={() => handleTabChange('paste')}
          >
            <Text style={[styles.tabText, activeTab === 'paste' && styles.tabTextActive]}>
              Paste email
            </Text>
          </TouchableOpacity>
        </View>

        {/* Content area */}
        <View style={styles.content}>
          {activeTab === 'paste' ? (
            <TextInput
              style={[styles.textInput, styles.softShadow]}
              multiline
              placeholder="Paste email..."
              placeholderTextColor="#5f6368"
              value={pastedText}
              onChangeText={handlePasteChange}
              editable={!isBusy}
              textAlignVertical="top"
            />
          ) : (
            <ScrollView style={styles.emailList} showsVerticalScrollIndicator={false}>
              {TEST_EMAILS.map((email) => (
                <TouchableOpacity
                  key={email.id}
                  style={[
                    styles.emailItem,
                    styles.softShadow,
                    selectedEmailId === email.id && styles.emailItemSelected,
                  ]}
                  disabled={isBusy}
                  onPress={() => handleEmailSelect(email.id)}
                >
                  <Text style={styles.emailLabel}>{email.label}</Text>
                  <Text style={styles.emailPreview} numberOfLines={1}>
                    {email.preview}
                  </Text>
                </TouchableOpacity>
              ))}
            </ScrollView>
          )}
        </View>

        {(activeTab === 'select' && selectedEmail) || isEncrypting || ciphertext ? (
          <View style={[styles.selectedEmailPreview, styles.softShadow]}>
            <Text style={styles.selectedEmailTitle}>{ciphertext ? 'Ciphertext' : 'Full email'}</Text>
            {isEncrypting ? (
              <View style={styles.loadingState}>
                <ActivityIndicator size="small" color="#8e8e93" />
                <Text style={styles.loadingText}>Encrypting email…</Text>
              </View>
            ) : ciphertext ? (
              <>
                <ScrollView
                  style={styles.ciphertextScroll}
                  showsVerticalScrollIndicator
                  nestedScrollEnabled
                >
                  <Text selectable style={styles.ciphertextText}>{ciphertext}</Text>
                </ScrollView>
                {isClassifying && (
                  <View style={styles.loadingState}>
                    <ActivityIndicator size="small" color="#8e8e93" />
                    <Text style={styles.loadingText}>Waiting for classification…</Text>
                  </View>
                )}
                {prediction && (
                  <View style={styles.resultBlock}>
                    <Text style={styles.resultText}>
                      {prediction.classification === 'spam' ? 'SPAM' : 'HAM'} (score {prediction.score.toFixed(3)})
                    </Text>
                    <Text style={styles.resultDescription}>
                      {prediction.classification === 'spam'
                        ? 'this would NOT be sent to your inbox'
                        : 'this would be sent to your inbox'}
                    </Text>
                  </View>
                )}
              </>
            ) : selectedEmail ? (
              <ScrollView showsVerticalScrollIndicator nestedScrollEnabled>
                <Text style={styles.selectedEmailText}>{selectedEmail.preview}</Text>
              </ScrollView>
            ) : null}
          </View>
        ) : null}

        {/* Footer */}
        <View style={styles.footer}>
          {!prediction && (
            <TouchableOpacity
              style={[styles.encryptButton, styles.buttonShadow, isBusy && styles.encryptButtonDisabled]}
              activeOpacity={0.8}
              disabled={isBusy}
              onPress={handlePrimaryAction}
            >
              <Text style={styles.encryptButtonText}>
                {isEncrypting
                  ? `Encrypting${'.'.repeat(encryptingDots)}`
                  : ciphertext
                    ? 'Send ciphertext for classification'
                    : 'Encrypt'}
              </Text>
            </TouchableOpacity>
          )}
          {error && <Text style={styles.errorText}>{error}</Text>}
        </View>

      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f2f2f7',
    paddingHorizontal: 20,
    paddingBottom: 24,
  },
  hiddenWebView: {
    width: 0,
    height: 0,
  },
  softShadow: {
    shadowColor: '#1c1c1e',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.06,
    shadowRadius: 5,
    elevation: 1,
  },
  buttonShadow: {
    shadowColor: '#005fcc',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.14,
    shadowRadius: 6,
    elevation: 2,
  },
  aboutLink: {
    position: 'absolute',
    right: 20,
    zIndex: 1,
    fontSize: 16,
    fontFamily: 'Inter_600SemiBold',
    color: '#005fcc',
  },
  header: {
    width: '100%',
    maxWidth: 600,
    alignSelf: 'center',
    alignItems: 'center',
    marginBottom: 24,
  },
  title: {
    width: '100%',
    fontSize: 32,
    fontFamily: 'Inter_700Bold',
    color: '#1c1c1e',
    letterSpacing: -0.5,
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 16,
    fontFamily: 'Inter_400Regular',
    color: '#5f6368',
    marginTop: 4,
    textAlign: 'center',
  },
  card: {
    flex: 1,
    width: '100%',
    maxWidth: 700,
    alignSelf: 'center',
    justifyContent: 'center',
    backgroundColor: 'transparent',
  },
  tabs: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 12,
    marginBottom: 18,
  },
  tab: {
    minHeight: 48,
    paddingVertical: 11,
    paddingHorizontal: 22,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 999,
    borderWidth: 1.5,
    borderColor: '#d1d1d6',
    backgroundColor: '#ffffff',
  },
  tabActive: {
    backgroundColor: '#005fcc',
    borderColor: '#005fcc',
  },
  tabText: {
    fontSize: 16,
    fontFamily: 'Inter_500Medium',
    lineHeight: 22,
    color: '#5f6368',
    fontWeight: '500',
  },
  tabTextActive: {
    color: '#ffffff',
    fontFamily: 'Inter_600SemiBold',
  },
  content: {
    marginBottom: 18,
  },
  textInput: {
    height: 260,
    borderWidth: 1,
    borderColor: '#e5e5ea',
    borderRadius: 12,
    padding: 16,
    fontSize: 16,
    fontFamily: 'Inter_400Regular',
    lineHeight: 24,
    color: '#1c1c1e',
    backgroundColor: '#fafafa',
  },
  emailList: {
    height: 260,
  },
  emailItem: {
    minHeight: 64,
    padding: 16,
    justifyContent: 'center',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#e5e5ea',
    marginBottom: 10,
    backgroundColor: '#fafafa',
  },
  emailItemSelected: {
    borderColor: '#007AFF',
    backgroundColor: '#EBF4FF',
  },
  emailLabel: {
    fontSize: 16,
    lineHeight: 22,
    fontFamily: 'Inter_600SemiBold',
    color: '#1c1c1e',
  },
  emailPreview: {
    fontSize: 14,
    lineHeight: 20,
    fontFamily: 'Inter_400Regular',
    color: '#5f6368',
    marginTop: 5,
  },
  selectedEmailPreview: {
    height: 220,
    marginBottom: 18,
    padding: 16,
    borderWidth: 1,
    borderColor: '#e5e5ea',
    borderRadius: 12,
    backgroundColor: '#ffffff',
  },
  selectedEmailTitle: {
    marginBottom: 10,
    fontSize: 14,
    fontFamily: 'Inter_600SemiBold',
    color: '#5f6368',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  selectedEmailText: {
    fontSize: 16,
    lineHeight: 24,
    fontFamily: 'Inter_400Regular',
    color: '#1c1c1e',
  },
  ciphertextScroll: {
    flex: 1,
  },
  ciphertextText: {
    fontSize: 13,
    lineHeight: 19,
    color: '#3a3a3c',
    fontFamily: Platform.select({ ios: 'Menlo', android: 'monospace', default: 'monospace' }),
  },
  loadingState: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
  },
  loadingText: {
    fontSize: 15,
    fontFamily: 'Inter_400Regular',
    color: '#5f6368',
  },
  footer: {
    alignItems: 'center',
    marginTop: 6,
  },
  encryptButton: {
    minHeight: 48,
    maxWidth: '100%',
    backgroundColor: '#005fcc',
    paddingHorizontal: 20,
    paddingVertical: 14,
    borderRadius: 24,
    justifyContent: 'center',
  },
  encryptButtonText: {
    color: '#ffffff',
    fontSize: 17,
    lineHeight: 22,
    fontFamily: 'Inter_600SemiBold',
    textAlign: 'center',
  },
  encryptButtonDisabled: {
    opacity: 0.6,
  },
  resultText: {
    fontSize: 18,
    fontFamily: 'Inter_700Bold',
    color: '#1c1c1e',
  },
  resultBlock: {
    marginTop: 10,
    alignItems: 'center',
  },
  resultDescription: {
    marginTop: 3,
    fontSize: 14,
    fontFamily: 'Inter_400Regular',
    color: '#5f6368',
    textAlign: 'center',
  },
  errorText: {
    marginTop: 12,
    maxWidth: 420,
    textAlign: 'center',
    fontSize: 14,
    lineHeight: 20,
    fontFamily: 'Inter_400Regular',
    color: '#c00',
  },
});
