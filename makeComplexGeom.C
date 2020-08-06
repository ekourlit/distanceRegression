
// makes a sensor volume of some type and maximal extension LxLxL
TGeoVolume* makeSensor(double L, TGeoMedium* medium, int type = 0) {
  auto geom = gGeoManager;
  if (!geom) {
    return nullptr;
  }
  // a simple box
  if (type == 0) {
    return geom->MakeBox("SENSOR", medium, L, L, L);
  }
  // a small polycone with 3 sections
  else if (type == 1) {
    const int nz = 3;
    auto pconvol = geom->MakePcon("SENSOR", medium, 0, 360, nz);
    const auto rmin = 0.;
    auto pcon = static_cast<TGeoPcon*>(pconvol->GetShape());
    pcon->DefineSection(0, -L/2, rmin, L/2);
    pcon->DefineSection(1, 0, rmin, 2.*L/3.);
    pcon->DefineSection(2, L/2, rmin, L/2);
    return pconvol;
  }
  // do an Xtruded solid (typical for ALICE)
  else if (type == 2) {
    return nullptr;
  }
  // do a simple tube
  else if (type == 3) {
    return geom->MakeTubs("SENSOR", medium, 0., L, L, 0, 180); 
  }
  return nullptr;
}

// hook the sensors into a mother
// we will make many sensors on the surface of a cube of halflength L and some given spacing
void attachSensors(TGeoVolume *mother, TGeoVolume *sensor, int NSensors, double L)
{
  int nodecounter = 0;
  // face y-x at x=-L and x=+L
  const double delta = 2 * L / NSensors;
  for (int nx = 0; nx < NSensors; ++nx) {
    const auto x = -L + delta * (nx + 1. / 2);
    for (int ny = 0; ny < NSensors; ++ny) {
      const auto y = -L + delta * (ny + 1. / 2);
      auto tr1     = new TGeoTranslation(L, x, y);

      mother->AddNode(sensor, nodecounter++, tr1);
      auto tr2 = new TGeoTranslation(-L, x, y);
      mother->AddNode(sensor, nodecounter++, tr2);
    }
  }
  // face x-z at y=-L and y=+L
  for (int nx = 0; nx < NSensors; ++nx) {
    const auto x = -L + delta * (nx + 1. / 2);
    for (int nz = 0; nz < NSensors; ++nz) {
      const auto z = -L + delta * (nz + 1. / 2);
      auto tr1     = new TGeoTranslation(x, -L, z);
      mother->AddNode(sensor, nodecounter++, tr1);
      auto tr2 = new TGeoTranslation(x, L, z);
      mother->AddNode(sensor, nodecounter++, tr2);
    }
  }
  // face x-y at z=-L and z=+L
  for (int nx = 0; nx < NSensors; ++nx) {
    const auto x = -L + delta * (nx + 1. / 2);
    for (int ny = 0; ny < NSensors; ++ny) {
      const auto y = -L + delta * (ny + 1. / 2);
      auto tr1     = new TGeoTranslation(x, y, -L);
      mother->AddNode(sensor, nodecounter++, tr1);
      auto tr2 = new TGeoTranslation(x, y, L);
      mother->AddNode(sensor, nodecounter++, tr2);
    }
  }
}

void makeComplexGeom(int type = 1, int NSensor = 2, bool dense = true) {
/// simple macro making a simple geometry yet scalable geometry
/// to test the effect of voxelization
  auto geom = new TGeoManager("simple1", "Simple geometry");

  //***************************************************************//
  //                          Elements
  //***************************************************************//
  TGeoElementTable *elTable = gGeoManager -> GetElementTable();
  TGeoElement *elAr = elTable -> GetElement(18); //!< Argon
  auto elKr = elTable->GetElement(36); //!< Krypton

  /**
   Argon Gas
   * density of argon gas at 20`C : 1.634e-03 [g/cm3]   (ref: wolframalpha.com)
   **/
  Double_t denArgonGas = 1.782e-03;
  const double denKryptonGas = 3.749e-03;
  TGeoMaterial *matArgonGas = new TGeoMaterial("ArgonGas", elAr, denArgonGas);

  //--- define materials
  TGeoMaterial *matVacuum = new TGeoMaterial("Vacuum", 1, 1, 1E-10);
  auto matAl = new TGeoMaterial("Al", 26.98, 13, 2.7);
  auto matSensor = new TGeoMaterial("Kr", elKr, denKryptonGas); // for the sensor (a gas in order not to shower)

  //--- define some media
  TGeoMedium *Vacuum = new TGeoMedium("VacuumMed",1, matVacuum);
  TGeoMedium *Al = new TGeoMedium("AlumMedium",2, matAl);
  TGeoMedium *Argon = new TGeoMedium("ArgonMedium",3, matArgonGas);
  auto *Sensor = new TGeoMedium("SensorMedium", 4, matSensor);

  double L1 = 160.;
  double L2 = 120;
  double L3 = 110;

  // we will put NSensor x NSensor on each surface of a virtual cube
  // hence there will be NSensor x NSensor x 6 total sensors in the geometry

  double Lsensor = (L2 - L3) / 2;
  while (Lsensor * NSensor > L2) {
    Lsensor *= 0.95;
  }

  TGeoVolume *top = geom->MakeBox("TOP", Vacuum, L1, L1, L1);
  geom->SetTopVolume(top);
  TGeoVolume *densebox = dense ? geom->MakeBox("DENSE", Al, L2, L2, L2) : geom->MakeBox("DENSE", Sensor, L2, L2, L2);
  TGeoVolume *gasbox = geom->MakeBox("GAS", Argon, 60, 60, 60);

  // make sensor out of Polycon
  auto sensorvol = makeSensor(Lsensor, Sensor, type);
  attachSensors(densebox, sensorvol, NSensor, (L2 + L3) / 2.);

  // identity transformation
  TGeoTranslation *tr1 = new TGeoTranslation(0., 0, 0.);

  densebox->AddNode(gasbox, 1);
  top->AddNode(densebox, 1);

  gGeoManager->CloseGeometry();
  gGeoManager->LockGeometry();
  gGeoManager->Export("geom.root");
  gGeoManager->Export("geom.gdml");
}
