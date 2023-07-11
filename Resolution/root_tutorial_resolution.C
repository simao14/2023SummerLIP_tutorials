#include "TROOT.h"
#include "TH1.h"
#include "TTree.h"
#include "TH2.h"
#include "TF1.h"
#include "TFile.h"
#include "TMath.h"
#include "TSystem.h"

using namespace std;

using std::cout;
using std::endl;

void root_tutorial_resolution(){

    TFile *file(0);
   TString fname = "./tmva_class_example.root";
   if (!gSystem->AccessPathName( fname )) {
      file = TFile::Open( fname ); // check if file in local directory exists
   }
   else {
      TFile::SetCacheFileDir(".");
      file = TFile::Open("http://root.cern.ch/files/tmva_class_example.root", "CACHEREAD");
   }
   if (!file) {
      std::cout << "ERROR: could not open data file" << std::endl;
      exit(1);
   }

    TCanvas *c = new TCanvas("c","c",700,700);
    c->cd();

    TTree* TreeS = (TTree*) file->Get("TreeS");
    TTree* TreeB = (TTree*) file->Get("TreeB");

    TreeS->Draw("var1");

    c->SaveAs("var1_dist_hist.png");

    TCanvas *cpad = new TCanvas("c","c",700,700);
    cpad->cd();
    TPad *p1 = new TPad("p1","p1",0.,0.45,1.,1);
	p1->SetBorderMode(1); 
	p1->SetFrameBorderMode(0); 
	p1->SetBorderSize(2);
	p1->SetBottomMargin(0.10);
	p1->Draw(); 

	TPad *p2 = new TPad("p2","p2",0.,0.,1.,0.45); 
	p2->SetTopMargin(0.);    
	p2->SetBorderMode(0);
	p2->SetBorderSize(2); 
	p2->SetFrameBorderMode(0); 
	p2->SetTicks(1,1); 
	p2->Draw();

    p1->cd();

    TreeS->Draw("var1");

    p2->cd();

    TreeB->Draw("var1");   

    cpad->SaveAs("var1_dist_pads_hist.png"); 

    TCanvas *ccut = new TCanvas("c","c",700,700);
    ccut->cd();

    TreeS->Draw("var1","var1>0 || var3<var4");

    ccut->SaveAs("var1_cut_hist.png");

    TFile * fout= new TFile("tom.root","RECREATE");
    fout->cd();

    int nevents =TreeS->GetEntries();
    double newvar[nevents];
    TTree* TreeS_2 = (TTree*) TreeS->Clone("TreeS");
    TTree* TreeB_2 = (TTree*) TreeS->Clone("TreeB");
    TBranch* nv = TreeS_2->Branch("newvar",newvar);

    Float_t var1[1];
    Float_t var2[1];
    TreeS->SetBranchAddress("var1",var1);
	TreeS->SetBranchAddress("var2",var2);
    for (int i=0;i<nevents;i++){
        TreeS->GetEntry(i);
        newvar[i]= TMath::Sqrt(var1[0]*var1[0] + var2[0]*var2[0]); 
        nv->Fill();
    }

    TreeS_2->Print();
    TreeS_2->Write();
    TreeB_2->Write();
    fout->Close();
    file->Close();
}