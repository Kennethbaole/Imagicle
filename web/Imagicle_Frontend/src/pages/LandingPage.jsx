import React, { useRef } from "react";
import { Link, useNavigate } from "react-router-dom";
import {
  MessageSquareDashed, Play, ChevronDown, CheckCircle, Linkedin,
  RotateCcw, BrushCleaning, Grip,
} from "lucide-react";
import TypewriterWord from "../pages/TypewriterWord";

import Kenneth from "../assets/Kenneth.jpeg";
import Camila from "../assets/Camila.jpeg";
import Kiara from "../assets/Kiara.jpeg";
import Cheryl from "../assets/Cheryl.jpeg";

/** Tiny placeholder avatar (initials + gradient) */
function Avatar({ name = "Member" }) {
  const initials =
    name
      .split(/\s+/)
      .map((p) => p[0]?.toUpperCase() || "")
      .slice(0, 2)
      .join("") || "??";
  return (
    <div
      className="w-24 h-24 rounded-full bg-gradient-to-br from-cyan-600 to-indigo-600
                        flex items-center justify-center text-white font-bold text-xl select-none"
    >
      {initials}
    </div>
  );
}

const LandingPage = () => {
  const appRef = useRef(null);
  const aboutUsRef = useRef(null);
  const navigate = useNavigate();

  const scrollToSection = (ref) =>
    ref.current?.scrollIntoView({ behavior: "smooth" });
  const handleHeadToApp = () => navigate("/app");

  /** How it Works cards, step-by-step */
  const features = [
    {
      step: "01",
      icon: <MessageSquareDashed className="w-8 h-8" />,
      title: "Type Your Prompt",
      description:
        'Enter a description in plain text, such as “a chair.”',
    },
    {
      step: "02",
      icon: <RotateCcw className="w-8 h-8" />,
      title: "Generate a Preview",
      description:
        "See your idea come to life as a dot-based 3D preview. The points float in space & form the rough of your object.",
    },
    {
      step: "03",
      icon: <BrushCleaning className="w-8 h-8" />,
      title: "Enhance the Result",
      description:
        "Upgrade to a denser and sharper point cloud, where thousands of points create clearer details.",
    },
    {
      step: "04",
      icon: <Grip className="w-8 h-8" />,
      title: "Make it Mesh (Optional)",
      description:
        "Transform the cloud of dots into a solid 3D model with surfaces and edges, ready to view, rotate, or export.",
    },
  ];

  /** Team list */
  const aboutUs = [
    { title: "Kenneth Le", role: "AI/ML Engineer", img: Kenneth, linkedin: "" },
    { title: "Camila Barbosa", role: "AI/ML Engineer", img: Camila, linkedin: "" },
    { title: "Kiara Delgado", role: "Backend Developer", img: Kiara, linkedin: "" },
    { title: "Cheryl Nguyen", role: "UI/UX Designer & Frontend Engineer", img: Cheryl, linkedin: "" },
  ];

  return (
    <div className="relative min-h-screen text-white">
        {/* BG image (behind everything) */}
    <div className="pointer-events-none absolute inset-0 -z-10 bg-[url('/particle-bg.jpg')] bg-cover bg-center bg-no-repeat" />

        {/* Dark overlay to reduce brightness / increase contrast */}
    <div className="pointer-events-none absolute inset-0 -z-10 bg-black/40" />

      {/* Navbar */}
      <nav className="fixed top-0 w-full bg-gray-950/90 backdrop-blur-sm z-50 border-b border-gray-700/50">
        <div className="relative flex items-center h-20 px-8 max-w-7xl mx-auto">
          {/* Brand */}
          <div className="absolute left-8">
            <button
              onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}
              className="text-2xl font-bold bg-gradient-to-r from-pink-700 via-pink-600 to-pink-500 bg-clip-text text-transparent
                            hover:from-cyan-300 hover:to-indigo-400 transition-all duration-300"
            >
              Imagicle
            </button>
          </div>

          {/* Center links */}
          <div className="flex-1 flex justify-center space-x-16">
            <button
              onClick={() => scrollToSection(appRef)}
              className="text-pink-600 hover:text-pink-600 transition-colors duration-300 font-medium"
            >
              What is Imagicle?
            </button>
            <button
              onClick={() => scrollToSection(aboutUsRef)}
              className="text-cyan-500 hover:text-pink-600 transition-colors duration-300 font-medium"
            >
              About Us
            </button>
          </div>

          {/* CTA */}
          <div className="absolute right-8">
            <button
              onClick={handleHeadToApp}
              className="px-8 py-3 bg-gradient-to-r from-pink-600 to-pink-500 text-white rounded-full font-semibold
                            transition-all duration-300 transform hover:scale-105 hover:shadow-lg hover:shadow-cyan-500/30"
            >
              Open App
            </button>
          </div>
        </div>
      </nav>

      {/* Landing Title */}
      <section
        ref={appRef}
        className="pt-36 pb-24 px-8"
    >

        <div className="max-w-7xl mx-auto text-center">
          {/* Project Title */}
          <h1 className="text-6xl md:text-7xl font-bold mb-12 font-bold bg-gradient-to-r from-pink-700 via-pink-600 to-pink-500 bg-clip-text text-transparent
                            hover:from-cyan-300 hover:to-indigo-400 transition-all duration-300 bg-clip-text text-transparent leading-tight">
            Imagicle
          </h1>

          {/* Hero Words */}
          <div className="flex flex-col items-center text-5xl md:text-5xl font-semibold text-white space-y-4">
            <TypewriterWord word="Type" speed={150} className="text-gray-300" />
            <span className="text-pink-500">See</span>
            <span className="text-pink-500">Shape</span>
          </div>

          {/* Tagline */}
          <p className="mt-8 text-xl md:text-3xl text-gray-300 text-center">
            Turn your ideas into three dimensional worlds!
          </p>

          {/* Spacer before features */}
          <div className="h-30"></div>

          {/* How it Works Title */}
          <div className="flex justify-center">
            <h2 className="text-6xl md:text-7xl font-bold bg-gradient-to-r from-pink-700 via-pink-600 to-pink-500 bg-clip-text text-transparent leading-tight">
              How it Works
            </h2>
          </div>
        </div>
      </section>

      {/* Features + Buttons wrapped together */}
      <section className="px-8 pb-16">
        {/* Features */}
        <div className="relative max-w-5xl mx-auto grid grid-cols-2 lg:grid-cols-2 gap-10 text-center justify-center">
          {features.map((feature, idx) => (
            <div
              key={idx}
              className="relative group bg-gradient-to-br from-gray-800/80 to-black/60 opacity-80 rounded-xl p-4 border border-gray-600/30 hover:border-cyan-400/40 transition-all duration-300 hover:scale-105 hover:shadow-xl hover:shadow-cyan-500/10"
            >
              <div className="text-pink-600 mb-8 text-4xl pb-3">{feature.icon}</div>
              <h3 className="text-2xl font-extrabold mb-5 text-cyan-600 pb-3">
                {feature.title}
              </h3>
              <p className="text-lg text-gray-300 group-hover:text-gray-200 transition-colors leading-relaxed pb-3">
                {feature.description}
              </p>
              <span className="absolute top-3 right-4 z-10 text-pink-500 text-xl font-bold opacity-70 pb-3">
                {String(idx + 1).padStart(2, "0")}
              </span>
            </div>
          ))}
        </div>

        {/* Buttons */}
        <div className="flex flex-col sm:flex-row gap-7 justify-center items-center mt-12 mb-[25px]">
          <Link
            to="/app"
            className="px-20 py-10 bg-gradient-to-r from-pink-700 to-pink-500 text-cyan-600 rounded-full text-2xl font-bold transition-all duration-300 transform hover:scale-105 hover:shadow-xl hover:shadow-cyan-500/20 flex items-center gap-3"
          >
            <Play className="w-10 h-10" />
            Try It Now
          </Link>
        </div>
      </section>

      {/* About Us (with real photos) */}
      <section ref={aboutUsRef} className="relative py-20 px-8 bg-black/40 overflow-hidden">

        {/* Decorative grid */}
        <div className="absolute inset-0">
          <div
            className="absolute inset-0 bg-[linear-gradient(rgba(59,130,246,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(59,130,246,0.03)_1px,transparent_1px)]
                            bg-[size:50px_50px]"
          />
          <div className="absolute top-0 left-1/4 w-px h-full bg-gradient-to-b from-transparent via-blue-500/20 to-transparent" />
          <div className="absolute top-0 right-1/3 w-px h-full bg-gradient-to-b from-transparent via-cyan-400/15 to-transparent" />
        </div>

        <div className="relative max-w-6xl mx-auto">
          <div className="mb-16 text-center">
            <h2
              className="text-5xl md:text-6xl font-bold mb-8 bg-gradient-to-r from-white via-pink-400 to-pink-600
                            bg-clip-text text-transparent leading-tight"
            >
              Meet Our Team
            </h2>
            <p className="text-xl md:text-4xl text-gray-300 mx-auto leading-relaxed mb-4">
              The builders behind <span className="font-semibold">imagicle</span>, crafting fast, beautiful 3D from text.
            </p>
          </div>

          {/* Team cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 lg:gap-12">
            {aboutUs.map((member, index) => (
              <div
                key={index}
                className="bg-black/70 backdrop-blur-sm border border-white/10 rounded-x1 p-6 flex flex-col items-center"
              >
                {/* Avatar/Image (changed to use member.img) */}
                <img
                  src={member.img}
                  alt={member.title}
                  className="w-56 h-56 rounded-full object-cover object-center ring-1 ring-white/10 shadow-md"
                />

                {/* Name */}
                <h3 className="mt-6 text-3xl font-extrabold text-pink-600 text-center">
                  {member.title}
                </h3>

                {/* Role */}
                <p className="mt-5 text-2xl text-white font-semibold text-center leading-snug">
                  {member.role}
                </p>

                {/* LinkedIns */}
                <a
                  href={member.linkedin || "#"}
                  onClick={(e) => {
                    if (!member.linkedin) e.preventDefault();
                  }}
                  target="_blank"
                  rel="noreferrer"
                  className={`mt-4 inline-flex items-center gap-2 px-4 py-2 rounded-md
                    ${member.linkedin ? "hover:bg-white/5" : "opacity-60 cursor-default"}`}
                >
                  <span className="inline-flex items-center justify-center w-8 h-8 rounded-full bg-white text-black">
                    <Linkedin className="w-4 h-4" />
                  </span>
                  <span className="text-cyan-400 font-bold tracking-wide">
                    LINKEDIN
                  </span>
                </a>
              </div>
            ))}
          </div>

          {/* Accent */}
          <div className="mt-20 flex items-center justify-center">
            <div className="flex items-center gap-4">
              <div className="w-8 h-px bg-gradient-to-r from-transparent to-slate-600" />
              <div className="w-2 h-2 rounded-full bg-cyan-400/60" />
              <div className="w-8 h-px bg-gradient-to-l from-transparent to-slate-600" />
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default LandingPage;
