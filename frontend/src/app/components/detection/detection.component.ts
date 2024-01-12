import { Component } from '@angular/core';
import { FrdService } from '../../service/frd.service';
import { SnackbarService } from '../../service/snackbar.service';
import { ClassificationResult } from '../../model/classification-result';
import { Authenticity } from '../../model/authenticity';

@Component({
  selector: 'app-detection',
  templateUrl: './detection.component.html',
  styleUrl: './detection.component.css',
})
export class DetectionComponent {
  history: string[] = [];
  result: ClassificationResult | null = {
    result: Authenticity.REAL,
    input_text: 'Test',
    probability: 0.9453,
  };

  constructor(
    private frdService: FrdService,
    private snackbarService: SnackbarService,
  ) {}

  checkReview(review: string): void {
    this.frdService.checkReview(review).subscribe({
      next: (result) => {
        this.result = result;
      },
      error: (err) => {
        this.snackbarService.showError(err);
      },
    });
  }
}
