import { TestBed } from '@angular/core/testing';

import { SnackbarService, DEFAULT_ERROR_MESSAGE } from './snackbar.service';
import { MatSnackBar, MatSnackBarModule } from '@angular/material/snack-bar';
import { HttpErrorResponse } from '@angular/common/http';

describe('SnackbarService', () => {
  let service: SnackbarService;
  let snackBar: MatSnackBar;

  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [MatSnackBarModule],
      providers: [SnackbarService],
    });
    service = TestBed.inject(SnackbarService);
    snackBar = TestBed.inject(MatSnackBar);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('should open snackbar when showError is called', () => {
    // given
    const spy = spyOn(snackBar, 'open');
    const errorMessage = 'error';

    // when
    service.showError(errorMessage);

    // then
    expect(spy).toHaveBeenCalledOnceWith(
      errorMessage,
      'Close',
      jasmine.objectContaining({
        horizontalPosition: 'center',
        verticalPosition: 'top',
        duration: 5000,
        panelClass: ['snackbar-error'],
      }),
    );
  });

  it('should call showError with DEFAULT_ERROR when handleError is called', () => {
    // given
    const showErrorSpy = spyOn(service, 'showError');
    const httpError = new HttpErrorResponse({ status: 500 });

    // when
    service.handleError(httpError);

    // then
    expect(showErrorSpy).toHaveBeenCalledOnceWith(DEFAULT_ERROR_MESSAGE);
  });
});
