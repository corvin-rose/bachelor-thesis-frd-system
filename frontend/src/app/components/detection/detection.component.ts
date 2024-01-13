import { Component, ElementRef, OnInit, ViewChild } from '@angular/core';
import { FrdService } from '../../service/frd.service';
import { SnackbarService } from '../../service/snackbar.service';
import { ClassificationResult } from '../../model/classification-result';
import { HistoryService } from '../../service/history.service';
import {
  AbstractControl,
  FormControl,
  FormGroupDirective,
  NgForm,
  ValidationErrors,
  Validators,
} from '@angular/forms';
import { ErrorStateMatcher } from '@angular/material/core';

@Component({
  selector: 'app-detection',
  templateUrl: './detection.component.html',
  styleUrl: './detection.component.css',
})
export class DetectionComponent implements OnInit {
  history: ClassificationResult[] = [];
  result: ClassificationResult | null = null;
  loading: boolean = false;

  reviewFormControl: FormControl = new FormControl('', [
    Validators.required,
    this.reviewMinLengthValidator(),
    this.reviewMaxLengthValidator(),
  ]);
  errorMatcher: ReviewErrorStateMatcher = new ReviewErrorStateMatcher();

  @ViewChild('reviewInput') reviewInput: ElementRef | undefined;

  constructor(
    private frdService: FrdService,
    private snackbarService: SnackbarService,
    private historyService: HistoryService,
  ) {}

  ngOnInit(): void {
    this.history = this.historyService.loadFromHistory();
  }

  checkReview(review: string): void {
    console.log(this.reviewFormControl);
    if (!this.reviewFormControl.valid) {
      return;
    }

    this.loading = true;
    this.frdService.checkReview(review).subscribe({
      next: (result) => {
        this.result = result;
        const hist = this.historyService.saveToHistory(result);
        this.history.push(hist);
        this.loading = false;
      },
      error: (err) => {
        this.snackbarService.showError(err);
        this.loading = false;
      },
    });
  }

  historyClick(key: string): void {
    this.result = this.history
      .map((h) => h as any)
      .filter((h) => h.key == key)
      .pop();
    if (this.reviewInput) {
      this.reviewInput.nativeElement.value = this.result?.input_text;
    }
  }

  historyUpdated(): void {
    this.history = this.historyService.loadFromHistory();
  }

  reviewMinLengthValidator() {
    return (control: AbstractControl): ValidationErrors | null => {
      const value = control.value as string;
      if (value.length < 3) {
        return { minLength: true };
      }
      return null;
    };
  }

  reviewMaxLengthValidator() {
    return (control: AbstractControl): ValidationErrors | null => {
      const value = control.value as string;
      if (value.length > 512) {
        return { maxLength: true };
      }
      return null;
    };
  }
}

export class ReviewErrorStateMatcher implements ErrorStateMatcher {
  isErrorState(
    control: FormControl | null,
    form: FormGroupDirective | NgForm | null,
  ): boolean {
    const isSubmitted = form && form.submitted;
    return !!(
      control &&
      control.invalid &&
      (control.dirty || control.touched || isSubmitted)
    );
  }
}
