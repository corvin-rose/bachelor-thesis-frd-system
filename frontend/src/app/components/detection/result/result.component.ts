import { Component, Input } from '@angular/core';
import { ClassificationResult } from '../../../model/classification-result';
import { Authenticity } from '../../../model/authenticity';

@Component({
  selector: 'app-result',
  templateUrl: './result.component.html',
  styleUrl: './result.component.css',
})
export class ResultComponent {
  @Input() result: ClassificationResult | null = null;
  protected readonly Authenticity = Authenticity;

  resultProbability(): number {
    return this.round((this.result?.probability ?? 0) * 100);
  }

  isFake(): boolean {
    return this.result?.result == Authenticity.FAKE;
  }

  round(val: number): number {
    return Math.round(val * 100) / 100;
  }
}
