import { Injectable } from '@angular/core';
import { environment } from '../app.config';
import { Observable } from 'rxjs';
import { HttpClient } from '@angular/common/http';
import { ClassificationResult } from '../model/classification-result';

@Injectable({
  providedIn: 'root',
})
export class FrdService {
  private apiServerUrl = environment.apiBaseUrl;

  constructor(private http: HttpClient) {}

  public checkReview(review: string): Observable<ClassificationResult> {
    return this.http.post<ClassificationResult>(
      `${this.apiServerUrl}/frd/`,
      JSON.stringify(review),
    );
  }
}
