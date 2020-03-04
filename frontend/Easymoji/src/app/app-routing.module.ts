import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';
import { FraseComponent } from './components/frase/frase.component';

// Rutas que vamos a acceder, el home es el frase component y validamos que
// no se pueda acceder a una ruta que no exista
const routes: Routes = [
  { path: '', component: FraseComponent},
  { path: '**', pathMatch: 'full', redirectTo: '' }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
