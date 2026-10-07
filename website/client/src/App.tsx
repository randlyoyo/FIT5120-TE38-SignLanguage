import { Route, Routes } from "react-router-dom";
import { SiteHeader } from "./components/SiteHeader";
import { ConversationPage } from "./pages/ConversationPage";
import { HomePage } from "./pages/HomePage";
import { LearnedSignsPage } from "./pages/LearnedSignsPage";
import { ScenarioLearningPage } from "./pages/ScenarioLearningPage";
import { ScenarioListPage } from "./pages/ScenarioListPage";
import { ScenarioCategoryPage } from "./pages/ScenarioCategoryPage";
import { SignDetailPage } from "./pages/SignDetailPage";
import { SignLibraryPage } from "./pages/SignLibraryPage";
import { ToLearnPage } from "./pages/ToLearnPage";

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
        <Route path="/scenarios" element={<ScenarioListPage />} />
        <Route path="/scenarios/category/:id" element={<ScenarioCategoryPage />} />
        <Route path="/scenarios/:id" element={<ScenarioLearningPage />} />
        <Route path="/to-learn" element={<ToLearnPage />} />
        <Route path="/learned" element={<LearnedSignsPage />} />
        <Route path="/signs/:id" element={<SignDetailPage />} />
        <Route path="/conversation" element={<ConversationPage />} />
      </Routes>
    </>
  );
}

export default App;
