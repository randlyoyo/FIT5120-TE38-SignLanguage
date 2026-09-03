import { Route, Routes } from "react-router-dom";
import { SiteHeader } from "./components/SiteHeader";
import { HomePage } from "./pages/HomePage";
import { LearnedSignsPage } from "./pages/LearnedSignsPage";
import { SignDetailPage } from "./pages/SignDetailPage";
import { SignLibraryPage } from "./pages/SignLibraryPage";

function App() {
  return (
    <>
      <div className="edge-spine edge-spine-left" aria-hidden="true">
        <span>HandMirror</span>
      </div>
      <div className="edge-spine edge-spine-right" aria-hidden="true">
        <span>Auslan Sign Catalogue</span>
      </div>
      <SiteHeader />
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/library" element={<SignLibraryPage />} />
        <Route path="/learned" element={<LearnedSignsPage />} />
        <Route path="/signs/:id" element={<SignDetailPage />} />
      </Routes>
    </>
  );
}

export default App;
