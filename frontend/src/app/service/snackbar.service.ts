import { Injectable } from '@angular/core';
import { MatSnackBar } from '@angular/material/snack-bar';
import { HttpErrorResponse } from '@angular/common/http';

export const DEFAULT_ERROR_MESSAGE: string =
  'Ein technischer Fehler ist aufgetreten';

@Injectable({
  providedIn: 'root',
})
export class SnackbarService {
  constructor(private snackBar: MatSnackBar) {}

  handleError(httpError: HttpErrorResponse): void {
    if (httpError.error != undefined) {
      console.error(httpError.error);
    }
    this.showError(DEFAULT_ERROR_MESSAGE);
    console.error(DEFAULT_ERROR_MESSAGE);
  }

  showError(error: string): void {
    this.snackBar.open(error, 'Close', {
      horizontalPosition: 'center',
      verticalPosition: 'top',
      duration: 5000,
      panelClass: ['snackbar-error'],
    });
  }
}
