import { useState } from 'react';
import {
  Platform,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import WebView from 'react-native-webview';

import { classifyEmail, type Prediction } from '@/lib/sealClient';

// Placeholder emails — will be replaced with real test set
const TEST_EMAILS = [
  { id: '1', label: 'Email 1 — ham', preview: 'Hi, just following up on our meeting...' },
  { id: '2', label: 'Email 2 — spam', preview: 'Congratulations! You have been selected...' },
  { id: '3', label: 'Email 3 — ham', preview: 'The report is attached for your review...' },
  { id: '4', label: 'Email 4 — spam', preview: 'URGENT: Your account requires verification...' },
];

type Tab = 'paste' | 'select';

export default function Index() {
  const insets = useSafeAreaInsets();
  const [activeTab, setActiveTab] = useState<Tab>('paste');
  const [pastedText, setPastedText] = useState('');
  const [selectedEmailId, setSelectedEmailId] = useState<string | null>(null);
  const [isEncrypting, setIsEncrypting] = useState(false);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [error, setError] = useState<string | null>(null);

  const selectedEmail = TEST_EMAILS.find((email) => email.id === selectedEmailId);
  const emailText = activeTab === 'paste' ? pastedText : selectedEmail?.preview ?? '';

  async function handleEncrypt() {
    if (!emailText.trim()) {
      setError('Enter an email or select a test email first.');
      return;
    }
    if (Platform.OS !== 'web') {
      setError('The browser CKKS client is currently available on Expo web only.');
      return;
    }

    setIsEncrypting(true);
    setPrediction(null);
    setError(null);
    try {
      setPrediction(await classifyEmail(emailText));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Encrypted prediction failed.');
    } finally {
      setIsEncrypting(false);
    }
  }

  return (
    <View style={[styles.container, { paddingTop: insets.top + 16 }]}>
      {/* Hidden WebView keeps the SEAL WASM context warm on mobile */}
      {Platform.OS !== 'web' && (
        <WebView
          style={styles.hiddenWebView}
          source={{ html: '<html><body></body></html>' }}
        />
      )}

      <View style={styles.header}>
        <Text style={styles.title}>PrivMail</Text>
        <Text style={styles.subtitle}>A homomorphic encryption demo</Text>
      </View>

      <View style={styles.card}>

        {/* Tab switcher */}
        <View style={styles.tabs}>
          <TouchableOpacity
            style={[styles.tab, activeTab === 'paste' && styles.tabActive]}
            onPress={() => setActiveTab('paste')}
          >
            <Text style={[styles.tabText, activeTab === 'paste' && styles.tabTextActive]}>
              Paste email
            </Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.tab, activeTab === 'select' && styles.tabActive]}
            onPress={() => setActiveTab('select')}
          >
            <Text style={[styles.tabText, activeTab === 'select' && styles.tabTextActive]}>
              Select email
            </Text>
          </TouchableOpacity>
        </View>

        {/* Content area */}
        <View style={styles.content}>
          {activeTab === 'paste' ? (
            <TextInput
              style={styles.textInput}
              multiline
              placeholder="Paste email..."
              placeholderTextColor="#999"
              value={pastedText}
              onChangeText={setPastedText}
              textAlignVertical="top"
            />
          ) : (
            <ScrollView style={styles.emailList} showsVerticalScrollIndicator={false}>
              {TEST_EMAILS.map((email) => (
                <TouchableOpacity
                  key={email.id}
                  style={[
                    styles.emailItem,
                    selectedEmailId === email.id && styles.emailItemSelected,
                  ]}
                  onPress={() => setSelectedEmailId(email.id)}
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

        {/* Footer */}
        <View style={styles.footer}>
          <TouchableOpacity
            style={[styles.encryptButton, isEncrypting && styles.encryptButtonDisabled]}
            activeOpacity={0.8}
            disabled={isEncrypting}
            onPress={handleEncrypt}
          >
            <Text style={styles.encryptButtonText}>
              {isEncrypting ? 'Encrypting…' : 'Encrypt'}
            </Text>
          </TouchableOpacity>
          {prediction && (
            <Text style={styles.resultText}>
              {prediction.classification === 'spam' ? 'SPAM' : 'HAM'} (score {prediction.score.toFixed(3)})
            </Text>
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
  header: {
    alignItems: 'center',
    marginBottom: 24,
  },
  title: {
    fontSize: 30,
    fontWeight: '700',
    color: '#1c1c1e',
    letterSpacing: -0.5,
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 14,
    color: '#8e8e93',
    marginTop: 4,
    textAlign: 'center',
  },
  card: {
    flex: 1,
    justifyContent: 'center',
    backgroundColor: 'transparent',
  },
  tabs: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 10,
    marginBottom: 16,
  },
  tab: {
    paddingVertical: 8,
    paddingHorizontal: 20,
    alignItems: 'center',
    borderRadius: 999,
    borderWidth: 1.5,
    borderColor: '#d1d1d6',
  },
  tabActive: {
    backgroundColor: '#007AFF',
    borderColor: '#007AFF',
  },
  tabText: {
    fontSize: 14,
    color: '#8e8e93',
    fontWeight: '500',
  },
  tabTextActive: {
    color: '#ffffff',
    fontWeight: '600',
  },
  content: {
    marginBottom: 14,
  },
  textInput: {
    height: 220,
    borderWidth: 1,
    borderColor: '#e5e5ea',
    borderRadius: 10,
    padding: 12,
    fontSize: 14,
    color: '#1c1c1e',
    backgroundColor: '#fafafa',
  },
  emailList: {
    height: 220,
  },
  emailItem: {
    padding: 12,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#e5e5ea',
    marginBottom: 8,
    backgroundColor: '#fafafa',
  },
  emailItemSelected: {
    borderColor: '#007AFF',
    backgroundColor: '#EBF4FF',
  },
  emailLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1c1c1e',
  },
  emailPreview: {
    fontSize: 12,
    color: '#8e8e93',
    marginTop: 3,
  },
  footer: {
    alignItems: 'center',
    marginTop: 4,
  },
  encryptButton: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 28,
    paddingVertical: 13,
    borderRadius: 22,
  },
  encryptButtonText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '600',
  },
  encryptButtonDisabled: {
    opacity: 0.6,
  },
  resultText: {
    marginTop: 14,
    fontSize: 16,
    fontWeight: '700',
    color: '#1c1c1e',
  },
  errorText: {
    marginTop: 12,
    maxWidth: 360,
    textAlign: 'center',
    fontSize: 13,
    color: '#c00',
  },
});
