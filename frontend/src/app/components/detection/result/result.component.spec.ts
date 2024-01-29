import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ResultComponent } from './result.component';
import { By } from '@angular/platform-browser';
import { ClassificationResult } from '../../../model/classification-result';
import { Authenticity } from '../../../model/authenticity';
import { HttpClientTestingModule } from '@angular/common/http/testing';
import { MaterialModule } from '../../../material.module';
import { ReactiveFormsModule } from '@angular/forms';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';

describe('ResultComponent', () => {
  let component: ResultComponent;
  let fixture: ComponentFixture<ResultComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ResultComponent],
      imports: [
        HttpClientTestingModule,
        MaterialModule,
        ReactiveFormsModule,
        BrowserAnimationsModule,
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(ResultComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should display input text correctly', () => {
    // given
    const mockResult: ClassificationResult = {
      input_text: 'Fake review',
      result: Authenticity.FAKE,
      probability: 0.95,
    };

    // when
    component.result = mockResult;
    fixture.detectChanges();

    const reviewDisplay = fixture.debugElement.query(
      By.css('em'),
    ).nativeElement;

    // then
    expect(reviewDisplay.textContent).toContain(mockResult.input_text);
  });

  it('should display and calculate probabilities correctly', () => {
    // given
    const mockResult1: ClassificationResult = {
      input_text: 'Fake review',
      result: Authenticity.FAKE,
      probability: 0.95,
    };
    const mockResult2: ClassificationResult = {
      input_text: 'Real review',
      result: Authenticity.REAL,
      probability: 0.85,
    };

    // when
    component.result = mockResult1;
    fixture.detectChanges();

    const primaryProgressBar = fixture.debugElement.query(
      By.css('.bar #primary-bar'),
    ).nativeElement;
    const secondaryProgressBar = fixture.debugElement.query(
      By.css('.bar #secondary-bar'),
    ).nativeElement;

    const primaryPercentage = fixture.debugElement.query(
      By.css('#primary-hint'),
    ).nativeElement;
    const secondaryPersentage = fixture.debugElement.query(
      By.css('#secondary-hint'),
    ).nativeElement;

    const reviewDisplay = fixture.debugElement.query(
      By.css('em'),
    ).nativeElement;

    // then
    expect(+primaryProgressBar.getAttribute('ng-reflect-value')).toEqual(95);
    expect(+secondaryProgressBar.getAttribute('ng-reflect-value')).toEqual(5);
    expect(primaryPercentage.textContent).toContain('95%');
    expect(secondaryPersentage.textContent).toContain('5%');
    expect(reviewDisplay.getAttribute('class')).toContain('fake');

    // and when
    component.result = mockResult2;
    fixture.detectChanges();

    // then
    expect(+primaryProgressBar.getAttribute('ng-reflect-value')).toEqual(85);
    expect(+secondaryProgressBar.getAttribute('ng-reflect-value')).toEqual(15);
    expect(primaryPercentage.textContent).toContain('85%');
    expect(secondaryPersentage.textContent).toContain('15%');
    expect(reviewDisplay.getAttribute('class')).toContain('real');
  });
});
