import { ComponentFixture, TestBed } from '@angular/core/testing';

import { DetectionComponent } from './detection.component';
import { HttpClientTestingModule } from '@angular/common/http/testing';
import { HistoryComponent } from './history/history.component';
import { MaterialModule } from '../../material.module';
import { ResultComponent } from './result/result.component';
import { ReactiveFormsModule } from '@angular/forms';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { FrdService } from '../../service/frd.service';
import { SnackbarService } from '../../service/snackbar.service';
import { HistoryService } from '../../service/history.service';
import { of, throwError } from 'rxjs';
import { ClassificationResult } from '../../model/classification-result';
import { HttpErrorResponse } from '@angular/common/http';
import { Authenticity } from '../../model/authenticity';
import { By } from '@angular/platform-browser';

describe('DetectionComponent', () => {
  let component: DetectionComponent;
  let fixture: ComponentFixture<DetectionComponent>;
  let frdServiceSpy: jasmine.SpyObj<FrdService>;
  let snackbarServiceSpy: jasmine.SpyObj<SnackbarService>;
  let historyServiceSpy: jasmine.SpyObj<HistoryService>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [DetectionComponent, HistoryComponent, ResultComponent],
      imports: [
        HttpClientTestingModule,
        MaterialModule,
        ReactiveFormsModule,
        BrowserAnimationsModule,
      ],
      providers: [
        {
          provide: FrdService,
          useValue: jasmine.createSpyObj('FrdService', ['checkReview']),
        },
        {
          provide: SnackbarService,
          useValue: jasmine.createSpyObj('SnackbarService', ['handleError']),
        },
        {
          provide: HistoryService,
          useValue: jasmine.createSpyObj('HistoryService', [
            'loadFromHistory',
            'saveToHistory',
          ]),
        },
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(DetectionComponent);
    component = fixture.componentInstance;

    frdServiceSpy = TestBed.inject(FrdService) as jasmine.SpyObj<FrdService>;
    snackbarServiceSpy = TestBed.inject(
      SnackbarService,
    ) as jasmine.SpyObj<SnackbarService>;
    historyServiceSpy = TestBed.inject(
      HistoryService,
    ) as jasmine.SpyObj<HistoryService>;
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should load history from HistoryService on initialization', () => {
    // given
    const mockHistory: ClassificationResult[] = [
      {
        input_text: 'Test review',
        result: Authenticity.FAKE,
        probability: 0.98,
      },
    ];
    historyServiceSpy.loadFromHistory.and.returnValue(mockHistory);

    // when
    fixture.detectChanges();

    // then
    expect(historyServiceSpy.loadFromHistory).toHaveBeenCalled();
    expect(component.history).toEqual(mockHistory);
  });

  it('should call checkReview and update history on successful review check', () => {
    // given
    const mockReview = 'Test review';
    const mockResult: ClassificationResult = {
      input_text: mockReview,
      result: Authenticity.FAKE,
      probability: 0.98,
    };
    frdServiceSpy.checkReview.and.returnValue(of(mockResult));
    component.reviewFormControl.setValue(mockReview);

    // when
    fixture.detectChanges();
    fixture.debugElement.query(By.css('#detect-button')).nativeElement.click();

    // then
    expect(frdServiceSpy.checkReview).toHaveBeenCalledWith(mockReview);
    expect(component.result).toEqual(mockResult);
    expect(historyServiceSpy.saveToHistory).toHaveBeenCalledWith(mockResult);
  });

  it('should handle error from frdService and show snack bar', () => {
    // given
    const mockError = new HttpErrorResponse({ status: 500 });
    frdServiceSpy.checkReview.and.returnValue(throwError(mockError));
    component.reviewFormControl.setValue('review');

    // when
    fixture.detectChanges();
    fixture.debugElement.query(By.css('#detect-button')).nativeElement.click();

    // then
    expect(snackbarServiceSpy.handleError).toHaveBeenCalledWith(mockError);
  });
});
