import React, { useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Image,
  NativeModules,
  Pressable,
  SafeAreaView,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { launchImageLibrary } from 'react-native-image-picker';

const { FaceRecognition } = NativeModules;

// ---------------------------------------------------------------------------
// Tipos
// ---------------------------------------------------------------------------

type Photo = {
  uri: string;
  embedding: number[] | null;
};

type Result = {
  distance: number;
  match: boolean;
};

// ---------------------------------------------------------------------------
// Componente principal
// ---------------------------------------------------------------------------

export default function App() {
  const [photo1, setPhoto1] = useState<Photo | null>(null);
  const [photo2, setPhoto2] = useState<Photo | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<Result | null>(null);

  const pickAndProcess = async (slot: 1 | 2) => {
    const response = await launchImageLibrary({ mediaType: 'photo' });
    if (response.didCancel || !response.assets?.[0]?.uri) return;

    const uri = response.assets[0].uri;
    setLoading(true);
    setResult(null);

    try {
      const embedding: number[] = await FaceRecognition.generateEmbedding(uri);
      const photo: Photo = { uri, embedding };
      if (slot === 1) setPhoto1(photo);
      else setPhoto2(photo);
    } catch (error: any) {
      Alert.alert('Erro ao processar imagem', error.message ?? 'Nenhum rosto detectado.');
    } finally {
      setLoading(false);
    }
  };

  const compare = async () => {
    if (!photo1?.embedding || !photo2?.embedding) return;
    setLoading(true);
    try {
      const distance: number = await FaceRecognition.compareEmbeddings(
        photo1.embedding,
        photo2.embedding,
      );
      setResult({ distance, match: distance < 0.72 });
    } catch (error: any) {
      Alert.alert('Erro ao comparar', error.message);
    } finally {
      setLoading(false);
    }
  };

  // Botão "Comparar" só aparece quando as duas fotos têm embedding gerado
  const canCompare = !!photo1?.embedding && !!photo2?.embedding;

  // ---------------------------------------------------------------------------
  // UI
  // ---------------------------------------------------------------------------

  return (
    <SafeAreaView style={styles.container}>

      {/* Cabeçalho */}
      <Text style={styles.title}>FamilyLens</Text>
      <Text style={styles.subtitle}>Prova de Conceito — Reconhecimento Facial</Text>

      {/* Slots das duas fotos lado a lado */}
      <View style={styles.photosRow}>
        <PhotoSlot
          label="Foto 1"
          photo={photo1}
          onPress={() => pickAndProcess(1)}
          disabled={loading}
        />
        <PhotoSlot
          label="Foto 2"
          photo={photo2}
          onPress={() => pickAndProcess(2)}
          disabled={loading}
        />
      </View>

      {/* Spinner de loading */}
      {loading && (
        <ActivityIndicator style={styles.spinner} size="large" color="#4A90E2" />
      )}

      {/* Botão Comparar — só renderiza quando as duas fotos estão prontas */}
      {canCompare && !loading && (
        <Pressable style={styles.compareButton} onPress={compare}>
          <Text style={styles.compareButtonText}>Comparar rostos</Text>
        </Pressable>
      )}

      {/* Resultado */}
      {result && !loading && (
        <ResultCard result={result} />
      )}

    </SafeAreaView>
  );
}

// ---------------------------------------------------------------------------
// Componente PhotoSlot
// Exibe um card clicável: mostra "+" se não tiver foto, ou a imagem se tiver.
// ---------------------------------------------------------------------------

type PhotoSlotProps = {
  label: string;
  photo: Photo | null;
  onPress: () => void;
  disabled: boolean;
};

function PhotoSlot({ label, photo, onPress, disabled }: PhotoSlotProps) {
  return (
    <View style={styles.slotWrapper}>
      <Text style={styles.slotLabel}>{label}</Text>
      {/* Pressable com opacidade reduzida quando disabled */}
      <Pressable
        style={[styles.slot, disabled && styles.slotDisabled]}
        onPress={onPress}
        disabled={disabled}>
        {photo ? (
          // Se tiver foto, exibe ela ocupando todo o slot
          <Image source={{ uri: photo.uri }} style={styles.slotImage} />
        ) : (
          // Se não tiver foto, exibe o ícone de adicionar
          <Text style={styles.slotPlaceholder}>+</Text>
        )}
      </Pressable>
      {/* Indicador embaixo do card: aguardando, processando ou pronto */}
      <Text style={styles.slotStatus}>
        {photo?.embedding ? '✅ Pronto' : photo ? '⏳ Processando...' : '—'}
      </Text>
    </View>
  );
}

// ---------------------------------------------------------------------------
// Componente ResultCard
// Exibe o resultado da comparação com cor e ícone de acordo com o veredito.
// ---------------------------------------------------------------------------

type ResultCardProps = {
  result: Result;
};

function ResultCard({ result }: ResultCardProps) {
  const isMatch = result.match;

  return (
    <View style={[styles.resultCard, isMatch ? styles.resultMatch : styles.resultNoMatch]}>
      <Text style={styles.resultIcon}>{isMatch ? '✅' : '❌'}</Text>
      <Text style={styles.resultVerdict}>
        {isMatch ? 'Mesma pessoa' : 'Pessoas diferentes'}
      </Text>
      <Text style={styles.resultDistance}>
        Distância cosseno: {result.distance.toFixed(4)}
      </Text>
      <Text style={styles.resultHint}>
        {isMatch
          ? 'Abaixo do threshold de 0.72 — os rostos são similares.'
          : 'Acima do threshold de 0.72 — os rostos são distintos.'}
      </Text>
    </View>
  );
}

// ---------------------------------------------------------------------------
// Estilos
// ---------------------------------------------------------------------------

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f0f2f5',
    alignItems: 'center',
    paddingTop: 24,
    paddingHorizontal: 16,
  },

  // Cabeçalho
  title: {
    fontSize: 26,
    fontWeight: 'bold',
    color: '#1a1a2e',
  },
  subtitle: {
    fontSize: 12,
    color: '#888',
    marginTop: 4,
    marginBottom: 32,
    textAlign: 'center',
  },

  // Linha com os dois slots
  photosRow: {
    flexDirection: 'row', // coloca os dois slots lado a lado
    gap: 16,
    marginBottom: 24,
  },

  // Wrapper de cada slot (label + card + status)
  slotWrapper: {
    alignItems: 'center',
  },
  slotLabel: {
    fontSize: 13,
    fontWeight: '600',
    color: '#555',
    marginBottom: 8,
  },

  // O card clicável em si
  slot: {
    width: 150,
    height: 150,
    borderRadius: 12,
    backgroundColor: '#fff',
    borderWidth: 2,
    borderColor: '#dde1e7',
    borderStyle: 'dashed',
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden', // garante que a imagem não ultrapasse o borderRadius
  },
  slotDisabled: {
    opacity: 0.5,
  },
  slotImage: {
    width: '100%',
    height: '100%',
  },
  slotPlaceholder: {
    fontSize: 40,
    color: '#bbb',
  },
  slotStatus: {
    marginTop: 6,
    fontSize: 12,
    color: '#888',
  },

  // Spinner
  spinner: {
    marginVertical: 16,
  },

  // Botão Comparar
  compareButton: {
    backgroundColor: '#4A90E2',
    paddingHorizontal: 40,
    paddingVertical: 14,
    borderRadius: 12,
    marginBottom: 24,
  },
  compareButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },

  // Card de resultado
  resultCard: {
    width: '100%',
    borderRadius: 16,
    padding: 20,
    alignItems: 'center',
    gap: 6,
  },
  resultMatch: {
    backgroundColor: '#e6f4ea',
    borderWidth: 1,
    borderColor: '#a8d5b5',
  },
  resultNoMatch: {
    backgroundColor: '#fdecea',
    borderWidth: 1,
    borderColor: '#f5c2be',
  },
  resultIcon: {
    fontSize: 36,
  },
  resultVerdict: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#1a1a1a',
  },
  resultDistance: {
    fontSize: 14,
    color: '#555',
  },
  resultHint: {
    fontSize: 12,
    color: '#888',
    textAlign: 'center',
    marginTop: 4,
  },
});
