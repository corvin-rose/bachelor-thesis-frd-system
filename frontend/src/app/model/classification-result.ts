import { Authenticity } from './authenticity';

export interface ClassificationResult {
  input_text: string;
  result: Authenticity;
  probability: number;
}
